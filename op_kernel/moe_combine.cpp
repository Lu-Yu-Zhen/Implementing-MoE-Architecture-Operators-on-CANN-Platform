// ============================================================================
// MoeCombine Kernel 实现
// ----------------------------------------------------------------------------
// 功能: Token 反重排与加权合并
// 输入: expert_out[numTokens, topK, hiddenSize] FP16,
//       expert_ids[numTokens, topK] INT32 (本 kernel 不直接使用),
//       weights[numTokens, topK] FP16
// 输出: y[numTokens, hiddenSize] FP16
// 算法: y[token, d] = Σ_k weights[token, k] * expert_out[token, k, d]
//
// 优化策略:
//   - 直接在 FP16 下做 Muls 和 Add，省掉 Cast 开销（combine 精度要求不高）
//   - 逐 token 处理，每次加载 topK 个专家输出并加权累加
//   - hiddenSize=2048 足够大，向量指令可充分流水
//
// UB 预算:
//   expertBuf: topK * hiddenSize * 2 = 2*2048*2 = 8KB（复用，每次只存一个专家的）
//   wBuf:      topK * 2 = 4B
//   yBuf:      hiddenSize * 2 = 4KB
//   总计: ~12KB << 256KB ✓
// ============================================================================

#include "kernel_operator.h"

using namespace AscendC;

// ============================================================================
// MoeCombine Kernel 算子类
// ============================================================================
class KernelMoeCombine {
public:
    __aicore__ inline KernelMoeCombine() {}

    __aicore__ inline void Init(GM_ADDR expertOut, GM_ADDR expertIds,
                                 GM_ADDR weights, GM_ADDR y, GM_ADDR tiling)
    {
        // 1. 解析 tiling 数据
        GET_TILING_DATA(tilingData, tiling);
        numTokens_    = tilingData.numTokens;
        hiddenSize_   = tilingData.hiddenSize;
        topK_         = tilingData.topK;
        tileS_        = tilingData.tileS;
        totalTaskNum_ = tilingData.totalTaskNum;

        // 2. 设置全局内存指针
        expertOutGm_.SetGlobalBuffer((__gm__ half*)expertOut,
                                      numTokens_ * topK_ * hiddenSize_);
        weightsGm_.SetGlobalBuffer((__gm__ half*)weights,
                                    numTokens_ * topK_);
        yGm_.SetGlobalBuffer((__gm__ half*)y,
                              numTokens_ * hiddenSize_);

        // 3. 分配 UB 缓冲区
        pipe_.InitBuffer(expertBuf_, hiddenSize_ * sizeof(half));    // 单个专家输出
        pipe_.InitBuffer(wBuf_, topK_ * sizeof(half));               // topK 个权重
        pipe_.InitBuffer(yBuf_, hiddenSize_ * sizeof(half));         // 累加结果
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = GetBlockIdx();
        const uint32_t coreNum = GetBlockNum();

        // 按 tileS 分块，多核轮询调度
        for (uint32_t taskId = coreId; taskId < totalTaskNum_; taskId += coreNum) {
            uint32_t tokenStart = taskId * tileS_;
            uint32_t tokenEnd = (tokenStart + tileS_ < numTokens_)
                                ? (tokenStart + tileS_) : numTokens_;

            for (uint32_t t = tokenStart; t < tokenEnd; t++) {
                ProcessOneToken(t);
            }
        }
    }

private:
    // ------------------------------------------------------------------
    // 处理单个 token 的加权合并
    // ------------------------------------------------------------------
    __aicore__ inline void ProcessOneToken(uint32_t tokenIdx)
    {
        // 1. 读取 weights[token, 0:topK] 到 UB
        auto wTensor = weightsGm_[tokenIdx * topK_];
        auto wLocal = wBuf_.template Get<half>();
        DataCopy(wLocal, wTensor, topK_);

        // 2. 初始化输出缓冲为 0（FP16 直接累加）
        auto yLocal = yBuf_.template Get<half>();
        Duplicate<half>(yLocal, (half)0.0f, hiddenSize_);

        // 3. 对每个专家 k，加权累加
        for (uint32_t k = 0; k < topK_; k++) {
            // 读取 expert_out[token, k, :] → expertBuf
            auto expertTensor = expertOutGm_[(tokenIdx * topK_ + k) * hiddenSize_];
            auto expertLocal = expertBuf_.template Get<half>();
            DataCopy(expertLocal, expertTensor, hiddenSize_);

            // 标量读取权重 weight[k]
            half w = wLocal.GetValue(k);

            // 加权: expertLocal *= weight[k]
            Muls<half>(expertLocal, expertLocal, w, hiddenSize_);

            // 累加: yLocal += expertLocal
            Add<half>(yLocal, yLocal, expertLocal, hiddenSize_);
        }

        // 4. 写出 y_buf 到 GM
        auto yTensor = yGm_[tokenIdx * hiddenSize_];
        DataCopy(yTensor, yLocal, hiddenSize_);
    }

    // ---- Tiling 参数 ----
    uint32_t numTokens_;
    uint32_t hiddenSize_;
    uint32_t topK_;
    uint32_t tileS_;
    uint32_t totalTaskNum_;

    // ---- 内存管理 ----
    TPipe pipe_;

    TBuf expertBuf_;   // 单个专家输出 [hiddenSize] FP16
    TBuf wBuf_;        // 权重 [topK] FP16
    TBuf yBuf_;        // 累加输出 [hiddenSize] FP16

    // ---- 全局内存 ----
    GlobalTensor<half> expertOutGm_;
    GlobalTensor<half> weightsGm_;
    GlobalTensor<half> yGm_;
};

// ============================================================================
// 算子核函数入口
// ============================================================================
extern "C" __global__ __aicore__ void MoeCombine(
    GM_ADDR expertOut, GM_ADDR expertIds, GM_ADDR weights,
    GM_ADDR y, GM_ADDR tiling)
{
    KernelMoeCombine op;
    op.Init(expertOut, expertIds, weights, y, tiling);
    op.Process();
}
