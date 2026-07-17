// ============================================================================
// MoeRoutingTopK Kernel 实现
// ----------------------------------------------------------------------------
// 功能: 门控路由 + Top-K 专家选择
// 输入: x[numTokens, hiddenSize] FP16, w_gate[hiddenSize, numExperts] FP16
// 输出: expert_ids[numTokens, topK] INT32, weights[numTokens, topK] FP16
// 算法: logits = x @ w_gate → softmax → TopK(K=2)
//
// 数据流:
//   1. 从 GM 搬入 x_tile[tileS, hiddenSize] 和 w_gate[hiddenSize, numExperts]
//   2. Cube 矩阵乘: logits[tileS, numExperts] = x_tile @ w_gate  (FP32 累加)
//   3. 逐行 Softmax: 数值稳定版 (减最大值 → exp → 归一化)
//   4. TopK(K=2): 两次扫描找最大/次大专家和权重
//   5. Cast FP32→FP16 后写回 GM
//
// UB 预算:
//   xBuf:       32*2048*2  = 128KB
//   wGateBuf:   2048*16*2  =  64KB (常驻)
//   logitsBuf:  32*16*4    =   2KB
//   temp:                  ≈  16KB
//   总计                   ≈ 210KB < 256KB ✓
// ============================================================================

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include <cstdlib>

using namespace AscendC;

// ============================================================================
// 编译期常量与类型别名
// ============================================================================
constexpr int32_t BUFFER_NUM = 2;       // 双缓冲，掩盖搬运延迟

// ============================================================================
// MatmulImpl 类型定义
// ============================================================================
// x[tileS, hiddenSize] @ w_gate[hiddenSize, numExperts] → logits[tileS, numExperts]
//   A: x       -- VECCALC (UB), ND, FP16, 不转置
//   B: w_gate  -- VECCALC (UB), ND, FP16, 不转置
//   C: logits  -- VECCALC (UB), ND, FP32
using gateAType = MatmulType<TPosition::VECCALC, CubeFormat::ND, half, false>;
using gateBType = MatmulType<TPosition::VECCALC, CubeFormat::ND, half, false>;
using gateCType = MatmulType<TPosition::VECCALC, CubeFormat::ND, float>;

// ============================================================================
// KernelMoeRouting 算子类
// ============================================================================
class KernelMoeRouting {
public:
    __aicore__ inline KernelMoeRouting() {}

    // ------------------------------------------------------------------
    // 初始化: 解析 Tiling、设置 GM 指针、分配 UB、初始化 MatmulImpl
    // ------------------------------------------------------------------
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR wGate,
                                 GM_ADDR expertIds, GM_ADDR weights,
                                 GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        InitParams(tilingData);

        // ---- 设置 GM 指针 ----
        const uint64_t xElemNum = static_cast<uint64_t>(numTokens_) * hiddenSize_;
        const uint64_t wElemNum = static_cast<uint64_t>(hiddenSize_) * numExperts_;
        const uint64_t idElemNum = static_cast<uint64_t>(numTokens_) * topK_;
        const uint64_t wtElemNum = static_cast<uint64_t>(numTokens_) * topK_;

        xGm_.SetGlobalBuffer((__gm__ half*)x, xElemNum);
        wGateGm_.SetGlobalBuffer((__gm__ half*)wGate, wElemNum);
        expertIdsGm_.SetGlobalBuffer((__gm__ int32_t*)expertIds, idElemNum);
        weightsGm_.SetGlobalBuffer((__gm__ half*)weights, wtElemNum);

        // ---- UB 缓冲区分配 ----
        // x 块: [tileS, hiddenSize] FP16
        pipe_.InitBuffer(xBuf_, tileS_ * hiddenSize_ * sizeof(half));

        // w_gate 常驻: [hiddenSize, numExperts] FP16
        pipe_.InitBuffer(wGateBuf_, hiddenSize_ * numExperts_ * sizeof(half));

        // logits: [tileS, numExperts] FP32 — GEMM 输出
        pipe_.InitBuffer(logitsBuf_, tileS_ * numExperts_ * sizeof(float));

        // softmax 结果缓冲（可与 logits 复用，但分开更清晰）
        pipe_.InitBuffer(softmaxBuf_, tileS_ * numExperts_ * sizeof(float));

        // TopK 索引: [tileS, topK] INT32
        pipe_.InitBuffer(topKIdsBuf_, tileS_ * topK_ * sizeof(int32_t));

        // TopK 权重: [tileS, topK] FP16 (最终输出)
        pipe_.InitBuffer(topKWBuf_, tileS_ * topK_ * sizeof(half));

        // 临时缓冲区: 用于向量操作 (exp, reduce 等)
        const uint32_t tempSize = (hiddenSize_ > numExperts_) ? hiddenSize_ : numExperts_;
        pipe_.InitBuffer(tempBuf_, tempSize * sizeof(float));

        // ---- MatmulImpl 初始化 ----
        // x[tileS, hiddenSize] @ w_gate[hiddenSize, numExperts] → logits[tileS, numExperts]
        // M=tileS, N=numExperts, K=hiddenSize
        InitCubeTiling(gateTilingBuf_, tileS_, numExperts_, hiddenSize_);
        TCubeTiling* gateTiling = reinterpret_cast<TCubeTiling*>(
            gateTilingBuf_.template Get<uint8_t>().GetPhyAddr());
        mmGate_.Init(gateTiling, &pipe_);
        mmGate_.SetSingleShape(tileS_, numExperts_, hiddenSize_);
    }

    // ------------------------------------------------------------------
    // 主处理循环: 按 taskId 分配 tile
    // ------------------------------------------------------------------
    __aicore__ inline void Process()
    {
        const uint32_t coreId = GetBlockIdx();
        const uint32_t coreNum = GetBlockNum();

        // 加载 w_gate 到 UB（常驻，所有 tile 共享）
        LoadWGate();

        // 动态分配 TopK 结果缓冲（每核一份，跨 tile 复用）
        int32_t* topKIds = (int32_t*)malloc(tileS_ * topK_ * sizeof(int32_t));
        half* topKWeights = (half*)malloc(tileS_ * topK_ * sizeof(half));

        for (uint32_t taskId = coreId; taskId < totalTaskNum_; taskId += coreNum) {
            ProcessOneTile(taskId, topKIds, topKWeights);
        }

        free(topKIds);
        free(topKWeights);
    }

private:
    // ------------------------------------------------------------------
    // 初始化 Tiling 参数
    // ------------------------------------------------------------------
    __aicore__ inline void InitParams(const MoeRoutingTilingData& tiling)
    {
        numTokens_    = tiling.numTokens;
        hiddenSize_   = tiling.hiddenSize;
        numExperts_   = tiling.numExperts;
        topK_         = tiling.topK;
        tileS_        = tiling.tileS;
        blockDim_     = tiling.blockDim;
        totalTaskNum_ = tiling.totalTaskNum;
        coreTaskNum_  = tiling.coreTaskNum;
    }

    // ------------------------------------------------------------------
    // 手动构造 TCubeTiling（参照 FlashAttention）
    // ------------------------------------------------------------------
    __aicore__ inline void InitCubeTiling(TBuf& buf, uint32_t M, uint32_t N, uint32_t K)
    {
        pipe_.InitBuffer(buf, sizeof(TCubeTiling));
        uint8_t* ptr = buf.template Get<uint8_t>().GetPhyAddr();
        TCubeTiling* t = reinterpret_cast<TCubeTiling*>(ptr);

        // 矩阵维度
        t->M  = static_cast<int32_t>(M);
        t->N  = static_cast<int32_t>(N);
        t->Ka = static_cast<int32_t>(K);
        t->Kb = static_cast<int32_t>(K);

        // 单核维度（单核处理整个 tile）
        t->singleCoreM = static_cast<int32_t>(M);
        t->singleCoreN = static_cast<int32_t>(N);
        t->singleCoreK = static_cast<int32_t>(K);

        // 基本块大小
        t->baseM = static_cast<int32_t>(M);
        t->baseN = static_cast<int32_t>(N);
        t->baseK = static_cast<int32_t>(K);

        // 步长与偏移
        t->stepM = 1;
        t->stepN = 1;
        t->stepKa = 1;
        t->stepKb = 1;

        // 循环次数与控制
        t->iterateOrder = 0;
        t->roundType = 0;
        t->isTransA = false;
        t->isTransB = false;
        t->doSpecialMulti = false;
        t->specialMultiType = 0;
        t->isGemmKLab = false;
        t->shareMode = 0;
        t->splitKMode = 0;
    }

    // ------------------------------------------------------------------
    // 加载 w_gate 到 UB（常驻，只需加载一次）
    // ------------------------------------------------------------------
    __aicore__ inline void LoadWGate()
    {
        LocalTensor<half> wLocal = wGateBuf_.Get<half>();
        DataCopy(wLocal, wGateGm_[0], hiddenSize_ * numExperts_);
    }

    // ------------------------------------------------------------------
    // 处理单个 tile: [taskId*tileS, min((taskId+1)*tileS, numTokens))
    // ------------------------------------------------------------------
    __aicore__ inline void ProcessOneTile(uint32_t taskId,
                                           int32_t* topKIds, half* topKWeights)
    {
        const uint32_t tokenStart = taskId * tileS_;
        const uint32_t tokenEnd = (tokenStart + tileS_ <= numTokens_) ?
                                   (tokenStart + tileS_) : numTokens_;
        const uint32_t actualTokens = tokenEnd - tokenStart;

        // Step 1: CopyIn — 加载 x_tile
        CopyInX(tokenStart, actualTokens);

        // Step 2: MatMul — logits = x_tile @ w_gate
        ComputeGate(actualTokens);

        // Step 3: Softmax — 逐行归一化
        SoftmaxRows(actualTokens);

        // Step 4: TopK — 找每行最大的 K 个专家和权重
        TopKSelect(actualTokens);

        // Step 5: CopyOut — 写回 expert_ids 和 weights
        CopyOut(tokenStart, actualTokens, topKIds, topKWeights);
    }

    // ------------------------------------------------------------------
    // CopyIn: 从 GM 加载 x_tile[tileS, hiddenSize] FP16
    // ------------------------------------------------------------------
    __aicore__ inline void CopyInX(uint32_t tokenStart, uint32_t actualTokens)
    {
        LocalTensor<half> xLocal = xBuf_.Get<half>();
        DataCopy(xLocal, xGm_[static_cast<uint64_t>(tokenStart) * hiddenSize_],
                 actualTokens * hiddenSize_);
    }

    // ------------------------------------------------------------------
    // MatMul: logits[tileS, numExperts] = x_tile @ w_gate  (Cube 单元)
    // ------------------------------------------------------------------
    __aicore__ inline void ComputeGate(uint32_t actualTokens)
    {
        LocalTensor<half> xLocal = xBuf_.Get<half>();
        LocalTensor<half> wLocal = wGateBuf_.Get<half>();
        LocalTensor<float> logitsLocal = logitsBuf_.Get<float>();

        // 设置尾块（最后一批 token 可能不满 tileS）
        if (actualTokens < tileS_) {
            mmGate_.SetTail(actualTokens, numExperts_);
        }

        // Cube 矩阵乘: logits = x_tile @ w_gate
        mmGate_.SetTensorA(xLocal);
        mmGate_.SetTensorB(wLocal);
        mmGate_.Iterate();
        mmGate_.GetTensorC(logitsLocal);
    }

    // ------------------------------------------------------------------
    // Softmax: 逐行做数值稳定的 softmax
    //   1. 找每行最大值
    //   2. 减去最大值（数值稳定）
    //   3. exp 取指数
    //   4. 求和并归一化
    // ------------------------------------------------------------------
    __aicore__ inline void SoftmaxRows(uint32_t actualTokens)
    {
        LocalTensor<float> logitsLocal = logitsBuf_.Get<float>();
        LocalTensor<float> softmaxLocal = softmaxBuf_.Get<float>();
        LocalTensor<float> tempRow = tempBuf_.Get<float>();

        for (uint32_t i = 0; i < actualTokens; i++) {
            const uint32_t rowOffset = i * numExperts_;

            // 1. 找每行最大值（数值稳定）
            float maxVal = logitsLocal.GetValue(rowOffset);
            for (uint32_t j = 1; j < numExperts_; j++) {
                float val = logitsLocal.GetValue(rowOffset + j);
                if (val > maxVal) {
                    maxVal = val;
                }
            }

            // 2. 减去最大值 → exp
            //    tempRow = logits[i] - maxVal
            Adds<float>(tempRow, logitsLocal[rowOffset], -maxVal, numExperts_);
            //    softmax[i] = exp(tempRow)
            Exp<float>(softmaxLocal[rowOffset], tempRow, numExperts_);

            // 3. 求和
            float sumVal = 0.0f;
            for (uint32_t j = 0; j < numExperts_; j++) {
                sumVal += softmaxLocal.GetValue(rowOffset + j);
            }

            // 4. 归一化: softmax[i] /= sum (除零保护)
            float invSum = (sumVal > 1e-30f) ? (1.0f / sumVal) : 0.0f;
            Muls<float>(softmaxLocal[rowOffset], softmaxLocal[rowOffset],
                        invSum, numExperts_);
        }
    }

    // ------------------------------------------------------------------
    // TopK (K=2): 对每行 softmax 结果找最大和次大的专家索引与权重
    //   简单两次扫描: 第一次找最大，第二次找次大（排除已选）
    // ------------------------------------------------------------------
    __aicore__ inline void TopKSelect(uint32_t actualTokens)
    {
        LocalTensor<float> softmaxLocal = softmaxBuf_.Get<float>();
        LocalTensor<int32_t> idsLocal = topKIdsBuf_.Get<int32_t>();
        LocalTensor<half> topKWLocal = topKWBuf_.Get<half>();
        LocalTensor<float> tempRow = tempBuf_.Get<float>();

        for (uint32_t i = 0; i < actualTokens; i++) {
            const uint32_t rowOffset = i * numExperts_;

            // --- 第一次扫描: 找最大 ---
            float bestVal = -1.0f;
            int32_t bestIdx = 0;
            for (uint32_t j = 0; j < numExperts_; j++) {
                float val = softmaxLocal.GetValue(rowOffset + j);
                if (val > bestVal) {
                    bestVal = val;
                    bestIdx = static_cast<int32_t>(j);
                }
            }

            // --- 第二次扫描: 找次大（排除 bestIdx）---
            float secondVal = -1.0f;
            int32_t secondIdx = 0;
            for (uint32_t j = 0; j < numExperts_; j++) {
                if (static_cast<int32_t>(j) == bestIdx) continue;
                float val = softmaxLocal.GetValue(rowOffset + j);
                if (val > secondVal) {
                    secondVal = val;
                    secondIdx = static_cast<int32_t>(j);
                }
            }

            // 写入 TopK 索引
            idsLocal.SetValue(i * topK_ + 0, bestIdx);
            idsLocal.SetValue(i * topK_ + 1, secondIdx);

            // 将 TopK 的 FP32 权重暂存到 temp，后续 Cast 为 FP16
            tempRow.SetValue(0, bestVal);
            tempRow.SetValue(1, secondVal);
        }

        // Cast: FP32 → FP16（批量转换所有 token 的 TopK 权重）
        // 先把所有权重收集到 tempBuf 的前 actualTokens*topK 位置
        // 由于 tempBuf 可能不够大，逐 token 处理
        for (uint32_t i = 0; i < actualTokens; i++) {
            const uint32_t rowOffset = i * numExperts_;

            // 重新获取权重（从 softmax 结果中按索引取值）
            int32_t idx0 = idsLocal.GetValue(i * topK_ + 0);
            int32_t idx1 = idsLocal.GetValue(i * topK_ + 1);
            float w0 = softmaxLocal.GetValue(rowOffset + idx0);
            float w1 = softmaxLocal.GetValue(rowOffset + idx1);

            // TopK 权重归一化: 使选中的 K 个权重之和为 1
            float wSum = w0 + w1;
            if (wSum > 0.0f) {
                w0 /= wSum;
                w1 /= wSum;
            }

            // 直接用 SetValue 写入 half 缓冲（隐式截断）
            topKWLocal.SetValue(i * topK_ + 0, static_cast<half>(w0));
            topKWLocal.SetValue(i * topK_ + 1, static_cast<half>(w1));
        }
    }

    // ------------------------------------------------------------------
    // CopyOut: 将 expert_ids 和 weights 写回 GM
    // ------------------------------------------------------------------
    __aicore__ inline void CopyOut(uint32_t tokenStart, uint32_t actualTokens,
                                    int32_t* topKIds, half* topKWeights)
    {
        LocalTensor<int32_t> idsLocal = topKIdsBuf_.Get<int32_t>();
        LocalTensor<half> topKWLocal = topKWBuf_.Get<half>();

        const uint64_t outOffset = static_cast<uint64_t>(tokenStart) * topK_;
        const uint32_t outElems = actualTokens * topK_;

        // 从 UB 拷贝到临时缓冲，再从临时缓冲写回 GM（保证 32 字节对齐）
        for (uint32_t i = 0; i < outElems; i++) {
            topKIds[i] = idsLocal.GetValue(i);
            topKWeights[i] = topKWLocal.GetValue(i);
        }

        // 写回 expert_ids [actualTokens, topK] INT32
        DataCopy(expertIdsGm_[outOffset], topKIds, outElems);

        // 写回 weights [actualTokens, topK] FP16
        DataCopy(weightsGm_[outOffset], topKWeights, outElems);
    }

private:
    // ---- Tiling 参数 ----
    uint32_t numTokens_;
    uint32_t hiddenSize_;
    uint32_t numExperts_;
    uint32_t topK_;
    uint32_t tileS_;
    uint32_t blockDim_;
    uint32_t totalTaskNum_;
    uint32_t coreTaskNum_;

    // ---- 内存管理 ----
    TPipe pipe_;

    TBuf xBuf_;          // x 块 [tileS, hiddenSize] FP16
    TBuf wGateBuf_;      // w_gate 常驻 [hiddenSize, numExperts] FP16
    TBuf logitsBuf_;     // GEMM 输出 [tileS, numExperts] FP32
    TBuf softmaxBuf_;    // softmax 结果 [tileS, numExperts] FP32
    TBuf topKIdsBuf_;    // TopK 索引 [tileS, topK] INT32
    TBuf topKWBuf_;      // TopK 权重 [tileS, topK] FP16
    TBuf tempBuf_;       // 临时缓冲区，用于向量操作

    // ---- MatmulImpl 对象 ----
    MatmulImpl<gateAType, gateBType, gateCType, gateCType> mmGate_;

    // TCubeTiling 缓冲区
    TBuf gateTilingBuf_;

    // ---- 全局内存 ----
    GlobalTensor<half> xGm_;
    GlobalTensor<half> wGateGm_;
    GlobalTensor<int32_t> expertIdsGm_;
    GlobalTensor<half> weightsGm_;
};

// ============================================================================
// 算子核函数入口
// ============================================================================
extern "C" __global__ __aicore__ void MoeRoutingTopK(
    GM_ADDR x, GM_ADDR wGate, GM_ADDR expertIds, GM_ADDR weights, GM_ADDR tiling)
{
    KernelMoeRouting op;
    op.Init(x, wGate, expertIds, weights, tiling);
    op.Process();
}
