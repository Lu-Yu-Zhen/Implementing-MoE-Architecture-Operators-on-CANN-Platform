// ============================================================================
// MoeExpertFFN Kernel — MoE 专家 FFN 前向计算（性能核心）
// ----------------------------------------------------------------------------
// 功能: 对每个专家执行 SwiGLU FFN:
//   gate = x @ W_gate^T          [tileTokens, intermediateSize]
//   up   = x @ W_up^T            [tileTokens, intermediateSize]
//   hidden = SiLU(gate) * up     [tileTokens, intermediateSize]
//   out  = hidden @ W_down       [tileTokens, hiddenSize]
//
// 数据类型: 输入/权重/输出 = FP16, GEMM 累加 = FP32
//
// 内存管理策略:
//   两阶段处理 + workspace (GM中间存储):
//   Stage A — SwiGLU: 对每个 iBlock 计算 gate/up/SiLU，写入 workspace
//   Stage B — Down:   从 workspace 读回，做 down_proj，写回 expert_out
//
//   K维(hiddenSize)分块: 每次处理 tileK=128，MatmulImpl 单次 Mmad
//   需要 K-blocking 循环: hiddenSize/tileK 次累加
//
// UB 预算 (~168KB < 256KB):
//   xBuf:        tileTokens * hiddenSize * 2        = 64KB
//   wTileBuf:    tileK * tileIntermediate * 2       = 64KB
//   projBuf:     tileTokens * tileIntermediate * 4  = 16KB (gate FP32, 复用于 up)
//   interBuf:    tileTokens * tileIntermediate * 2  = 8KB  (SwiGLU FP16)
//   tempBuf:     tileTokens * tileIntermediate * 4  = 16KB
//   tiling×2:                                       ≈ 2KB
//   总计:                                            ≈ 170KB
//
// workspace 大小: numExperts * ceil(maxTokensPerExpert/tileTokens) *
//                 intermediateSize * sizeof(half)
// ============================================================================

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

// ============================================================================
// 编译期常量
// ============================================================================
constexpr uint32_t MAX_TOKENS_PER_EXPERT = 512;  // 每专家最大 token 索引数
constexpr uint32_t MAX_TILE_BATCHES = 32;         // 每专家最大 tile 批次数

// ============================================================================
// MatmulImpl 类型定义
// ============================================================================
// Gate/Up 投影: A[MT, K] @ B[NI, K]^T → C[MT, NI]
//   A = x_block (UB), B = W_gate/up_tile (UB), C = 输出 (UB)
using ProjAType = MatmulType<TPosition::VECCALC, CubeFormat::ND, half, false>;
using ProjBType = MatmulType<TPosition::VECCALC, CubeFormat::ND, half, true>;
using ProjCType = MatmulType<TPosition::VECCALC, CubeFormat::ND, float>;

// Down 投影: A[MT, KI] @ B[KI, NH] → C[MT, NH]
//   A = intermediate (UB), B = W_down_tile (UB, 不转置), C = 输出 (UB)
using DownAType = MatmulType<TPosition::VECCALC, CubeFormat::ND, half, false>;
using DownBType = MatmulType<TPosition::VECCALC, CubeFormat::ND, half, false>;
using DownCType = MatmulType<TPosition::VECCALC, CubeFormat::ND, float>;

// ============================================================================
// MoeExpertFFN Kernel 算子类
// ============================================================================
class KernelMoeExpertFFN {
public:
    __aicore__ inline KernelMoeExpertFFN() {}

    // ====================================================================
    // 初始化: 解析 tiling → 设置 GM 指针 → 分配 UB → 初始化 MatmulImpl
    // ====================================================================
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR weights,
                                 GM_ADDR expertWeights, GM_ADDR expertOut,
                                 GM_ADDR workspace, GM_ADDR tiling)
    {
        // ---- 解析 tiling 参数 ----
        GET_TILING_DATA(tilingData, tiling);
        numTokens_       = tilingData.numTokens;
        hiddenSize_      = tilingData.hiddenSize;
        intermediateSize_= tilingData.intermediateSize;
        numExperts_      = tilingData.numExperts;
        topK_            = tilingData.topK;
        tileTokens_      = tilingData.tileTokens;
        tileIntermediate_= tilingData.tileIntermediate;
        tileK_           = tilingData.tileK;

        // ---- 设置 GM 指针 ----
        xGm_.SetGlobalBuffer((__gm__ half*)x,
            static_cast<uint64_t>(numTokens_) * hiddenSize_);
        expertIdsGm_.SetGlobalBuffer((__gm__ int32_t*)expertIds,
            static_cast<uint64_t>(numTokens_) * topK_);
        weightsGm_.SetGlobalBuffer((__gm__ half*)weights,
            static_cast<uint64_t>(numTokens_) * topK_);
        expertWeightsGm_.SetGlobalBuffer((__gm__ half*)expertWeights,
            static_cast<uint64_t>(numExperts_) * 3 * intermediateSize_ * hiddenSize_);
        expertOutGm_.SetGlobalBuffer((__gm__ half*)expertOut,
            static_cast<uint64_t>(numTokens_) * topK_ * hiddenSize_);
        workspaceGm_.SetGlobalBuffer((__gm__ half*)workspace,
            static_cast<uint64_t>(numExperts_) * MAX_TILE_BATCHES * tileTokens_ * intermediateSize_);

        // ---- UB 缓冲区分配 ----
        // x 数据: [tileTokens, hiddenSize] FP16 (Stage A 常驻)
        pipe_.InitBuffer(xBuf_, tileTokens_ * hiddenSize_ * sizeof(half));
        // 权重 tile: [tileK, tileIntermediate] FP16 (gate/up/down 共用)
        pipe_.InitBuffer(wTileBuf_, tileK_ * tileIntermediate_ * sizeof(half));
        // 投影输出: [tileTokens, tileIntermediate] FP32 (gate, 复用于 up)
        pipe_.InitBuffer(projBuf_, tileTokens_ * tileIntermediate_ * sizeof(float));
        // SwiGLU 中间结果: [tileTokens, tileIntermediate] FP16
        pipe_.InitBuffer(interBuf_, tileTokens_ * tileIntermediate_ * sizeof(half));
        // 向量化临时缓冲: [tileTokens, tileIntermediate] FP32
        pipe_.InitBuffer(tempBuf_, tileTokens_ * tileIntermediate_ * sizeof(float));

        // ---- MatmulImpl 初始化 ----
        // 投影 GEMM: M=tileTokens, N=tileIntermediate, K=tileK (K-blocking 手动)
        InitCubeTiling(projTilingBuf_, tileTokens_, tileIntermediate_, tileK_);
        TCubeTiling* projTiling = reinterpret_cast<TCubeTiling*>(
            projTilingBuf_.template Get<uint8_t>().GetPhyAddr());
        mmProj_.Init(projTiling, &pipe_);
        mmProj_.SetSingleShape(tileTokens_, tileIntermediate_, tileK_);

        // Down 投影: M=tileTokens, N=tileH(=tileIntermediate), K=tileIntermediate
        InitCubeTiling(downTilingBuf_, tileTokens_, tileIntermediate_, tileIntermediate_);
        TCubeTiling* downTiling = reinterpret_cast<TCubeTiling*>(
            downTilingBuf_.template Get<uint8_t>().GetPhyAddr());
        mmDown_.Init(downTiling, &pipe_);
        mmDown_.SetSingleShape(tileTokens_, tileIntermediate_, tileIntermediate_);
    }

    // ====================================================================
    // 主处理循环: 每核处理若干专家
    // ====================================================================
    __aicore__ inline void Process()
    {
        const uint32_t coreId = GetBlockIdx();
        const uint32_t coreNum = GetBlockNum();

        for (uint32_t expertId = coreId; expertId < numExperts_; expertId += coreNum) {
            ProcessOneExpert(expertId);
        }
    }

private:
    // ------------------------------------------------------------------
    // 手动构造 TCubeTiling (参照 FlashAttention)
    // ------------------------------------------------------------------
    __aicore__ inline void InitCubeTiling(TBuf& buf, uint32_t M, uint32_t N, uint32_t K)
    {
        pipe_.InitBuffer(buf, sizeof(TCubeTiling));
        uint8_t* ptr = buf.template Get<uint8_t>().GetPhyAddr();
        TCubeTiling* t = reinterpret_cast<TCubeTiling*>(ptr);

        t->M  = static_cast<int32_t>(M);
        t->N  = static_cast<int32_t>(N);
        t->Ka = static_cast<int32_t>(K);
        t->Kb = static_cast<int32_t>(K);
        t->singleCoreM = static_cast<int32_t>(M);
        t->singleCoreN = static_cast<int32_t>(N);
        t->singleCoreK = static_cast<int32_t>(K);
        t->baseM = static_cast<int32_t>(M);
        t->baseN = static_cast<int32_t>(N);
        t->baseK = static_cast<int32_t>(K);
        t->stepM = 1;
        t->stepN = 1;
        t->stepKa = 1;
        t->stepKb = 1;
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
    // 处理单个专家: 收集 token → Stage A → Stage B → Scatter
    // ------------------------------------------------------------------
    __aicore__ inline void ProcessOneExpert(uint32_t expertId)
    {
        // ---- Step 1: 扫描 expertIds，收集属于该专家的 (tokenIdx, slotIdx) ----
        // 使用 UB 中的小数组存储索引 (最多 MAX_TOKENS_PER_EXPERT 个)
        // 注: 此处用标量逐元素读取 expertIds，性能非瓶颈
        uint32_t tokenIndices[MAX_TOKENS_PER_EXPERT];
        uint32_t slotIndices[MAX_TOKENS_PER_EXPERT];
        uint32_t numAssigned = 0;

        for (uint32_t t = 0; t < numTokens_ && numAssigned < MAX_TOKENS_PER_EXPERT; t++) {
            for (uint32_t s = 0; s < topK_; s++) {
                int32_t eid = expertIdsGm_.GetValue(t * topK_ + s);
                if (eid == static_cast<int32_t>(expertId)) {
                    tokenIndices[numAssigned] = t;
                    slotIndices[numAssigned] = s;
                    numAssigned++;
                }
            }
        }

        if (numAssigned == 0) return;  // 该专家无分配 token，跳过

        // ---- 专家权重基地址 ----
        // expert_weights[numExperts, 3*intermediateSize, hiddenSize]
        // 专家 e: base = e * 3 * intermediateSize * hiddenSize
        const uint64_t expertWBase = static_cast<uint64_t>(expertId) * 3 *
                                      intermediateSize_ * hiddenSize_;

        // ---- workspace 基地址 (用于存放 intermediate 结果) ----
        // 每专家: MAX_TILE_BATCHES * tileTokens * intermediateSize 个元素
        const uint64_t wsBase = static_cast<uint64_t>(expertId) *
                                 MAX_TILE_BATCHES * tileTokens_ * intermediateSize_;

        // ---- Step 2: 按 tileTokens 分批处理 ----
        const uint32_t numBatches = (numAssigned + tileTokens_ - 1) / tileTokens_;

        for (uint32_t batchIdx = 0; batchIdx < numBatches; batchIdx++) {
            const uint32_t start = batchIdx * tileTokens_;
            const uint32_t actualRows = ((start + tileTokens_) <= numAssigned) ?
                                         tileTokens_ : (numAssigned - start);

            ProcessOneBatch(expertId, expertWBase, wsBase, batchIdx,
                           tokenIndices, slotIndices, start, actualRows);
        }
    }

    // ------------------------------------------------------------------
    // 处理一个 token 批次: Stage A (SwiGLU) + Stage B (Down proj)
    // ------------------------------------------------------------------
    __aicore__ inline void ProcessOneBatch(
        uint32_t expertId, uint64_t expertWBase, uint64_t wsBase,
        uint32_t batchIdx, const uint32_t* tokenIndices,
        const uint32_t* slotIndices, uint32_t start, uint32_t actualRows)
    {
        // ---- Gather: 从 x GM 中读取本批 token 数据 ----
        LocalTensor<half> xLocal = xBuf_.Get<half>();
        for (uint32_t i = 0; i < actualRows; i++) {
            uint32_t tokenIdx = tokenIndices[start + i];
            DataCopy(xLocal[i * hiddenSize_],
                     xGm_[tokenIdx * hiddenSize_], hiddenSize_);
        }

        // ---- Stage A: SwiGLU 计算，结果写入 workspace ----
        const uint64_t interWsOffset = wsBase +
            static_cast<uint64_t>(batchIdx) * tileTokens_ * intermediateSize_;
        ComputeSwiGLU(expertWBase, interWsOffset, actualRows);

        // ---- Stage B: Down projection，结果写入 workspace (FP16) ----
        const uint64_t downWsOffset = wsBase +
            static_cast<uint64_t>(batchIdx) * tileTokens_ * intermediateSize_;
        ComputeDownProj(expertWBase, downWsOffset, actualRows);

        // ---- Scatter: 从 workspace 读取结果，写入 expert_out GM ----
        for (uint32_t i = 0; i < actualRows; i++) {
            uint32_t tokenIdx = tokenIndices[start + i];
            uint32_t slotIdx = slotIndices[start + i];
            uint64_t outOffset = (static_cast<uint64_t>(tokenIdx) * topK_ + slotIdx)
                                  * hiddenSize_;
            DataCopy(expertOutGm_[outOffset],
                     workspaceGm_[downWsOffset + i * hiddenSize_], hiddenSize_);
        }
    }

    // ====================================================================
    // Stage A: SwiGLU 计算
    // 对每个 iBlock: gate = x @ W_gate^T, up = x @ W_up^T
    //                hidden = SiLU(gate) * up → 写入 workspace
    // ====================================================================
    __aicore__ inline void ComputeSwiGLU(uint64_t expertWBase,
                                          uint64_t wsOffset,
                                          uint32_t actualRows)
    {
        LocalTensor<half> xLocal = xBuf_.Get<half>();
        LocalTensor<half> wTile = wTileBuf_.Get<half>();
        LocalTensor<float> projOut = projBuf_.Get<float>();
        LocalTensor<half> interLocal = interBuf_.Get<half>();
        LocalTensor<float> tempLocal = tempBuf_.Get<float>();

        const uint32_t numIBlocks = intermediateSize_ / tileIntermediate_;
        const uint32_t numKBlocks = hiddenSize_ / tileK_;

        for (uint32_t iBlock = 0; iBlock < numIBlocks; iBlock++) {
            // W_gate 偏移: expert_base + iBlock * tileIntermediate * hiddenSize
            // W_gate 排布: [intermediateSize, hiddenSize]，行偏移 iBlock*tileIntermediate
            const uint64_t gateWOffset = expertWBase +
                static_cast<uint64_t>(iBlock) * tileIntermediate_ * hiddenSize_;

            // W_up 偏移: expert_base + (intermediateSize + iBlock*tileIntermediate) * hiddenSize
            const uint64_t upWOffset = expertWBase +
                (static_cast<uint64_t>(intermediateSize_) +
                 static_cast<uint64_t>(iBlock) * tileIntermediate_) * hiddenSize_;

            // ---- Gate 投影: gate_out = x @ W_gate_tile^T ----
            // K-blocking: 沿 hiddenSize 维分块累加
            Duplicate<float>(projOut, 0.0f, tileTokens_ * tileIntermediate_);

            for (uint32_t kBlock = 0; kBlock < numKBlocks; kBlock++) {
                // 加载 W_gate 的 K-tile: [tileK, tileIntermediate] → 转置加载
                // W_gate[tileIntermediate, hiddenSize] 的第 kBlock 列块:
                //   行 [0, tileIntermediate), 列 [kBlock*tileK, (kBlock+1)*tileK)
                // 因为 B 类型 isTransB=true，MatmulImpl 期望 B 排布为 [N, K]
                // 即 [tileIntermediate, tileK]，需从 GM 按列读取
                LoadWeightTile(wTile, gateWOffset, kBlock, true);

                // x 的 K-tile: [tileTokens, tileK]
                LocalTensor<half> xK = xLocal + kBlock * tileK_;

                // GEMM partial += xK @ W_gate_tile^T
                AccumulateGemm(xK, wTile, projOut, kBlock > 0);
            }

            // projBuf 现在包含 gate_out FP32 [tileTokens, tileIntermediate]
            // 对 gate 做 SiLU 并转 FP16 → interBuf
            ComputeSiLU(projOut, interLocal, tempLocal);

            // ---- Up 投影: up_out = x @ W_up_tile^T ----
            // 复用 projBuf 存放 up 输出
            Duplicate<float>(projOut, 0.0f, tileTokens_ * tileIntermediate_);

            for (uint32_t kBlock = 0; kBlock < numKBlocks; kBlock++) {
                LoadWeightTile(wTile, upWOffset, kBlock, true);
                LocalTensor<half> xK = xLocal + kBlock * tileK_;
                AccumulateGemm(xK, wTile, projOut, kBlock > 0);
            }

            // projBuf 现在包含 up_out FP32 [tileTokens, tileIntermediate]
            // hidden = SiLU(gate) * up → FP16
            // interLocal 已有 SiLU(gate) FP16，projOut 有 up FP32
            MulSiLU(interLocal, projOut, tempLocal);

            // 将 interLocal (FP16) 写入 workspace
            uint32_t writeCount = actualRows * tileIntermediate_;
            DataCopy(workspaceGm_[wsOffset +
                     static_cast<uint64_t>(iBlock) * tileIntermediate_],
                     interLocal, writeCount);
        }
    }

    // ====================================================================
    // Stage B: Down projection
    // 从 workspace 读回 intermediate，做 down_proj，结果覆写 workspace
    // ====================================================================
    __aicore__ inline void ComputeDownProj(uint64_t expertWBase,
                                            uint64_t wsOffset,
                                            uint32_t actualRows)
    {
        LocalTensor<half> xLocal = xBuf_.Get<half>();   // 复用为 intermediate 读缓冲
        LocalTensor<half> wTile = wTileBuf_.Get<half>();
        LocalTensor<float> downOut = projBuf_.Get<float>();  // 复用为 down 输出 FP32
        LocalTensor<half> interLocal = interBuf_.Get<half>();

        // W_down 基地址: expert_base + 2 * intermediateSize * hiddenSize
        // W_down 排布: [intermediateSize, hiddenSize] (转置存储)
        const uint64_t downWBase = expertWBase +
            2ULL * intermediateSize_ * hiddenSize_;

        const uint32_t numIBlocks = intermediateSize_ / tileIntermediate_;
        // down 输出沿 hidden 维分块，每次 tileIntermediate 列 (复用 tileIntermediate 作 tileH)
        const uint32_t numHBlocks = hiddenSize_ / tileIntermediate_;

        for (uint32_t hBlock = 0; hBlock < numHBlocks; hBlock++) {
            // 初始化 down 输出累加器为 0
            Duplicate<float>(downOut, 0.0f, tileTokens_ * tileIntermediate_);

            for (uint32_t iBlock = 0; iBlock < numIBlocks; iBlock++) {
                // 从 workspace 读回 intermediate tile [tileTokens, tileIntermediate] FP16
                uint32_t readCount = actualRows * tileIntermediate_;
                DataCopy(interLocal,
                         workspaceGm_[wsOffset +
                         static_cast<uint64_t>(iBlock) * tileIntermediate_],
                         readCount);

                // 加载 W_down tile: [tileIntermediate, tileIntermediate]
                // W_down[iBlock*tileIntermediate : (iBlock+1)*tileIntermediate,
                //          hBlock*tileIntermediate : (hBlock+1)*tileIntermediate]
                const uint64_t downTileOffset = downWBase +
                    static_cast<uint64_t>(iBlock) * tileIntermediate_ * hiddenSize_ +
                    static_cast<uint64_t>(hBlock) * tileIntermediate_;
                LoadDownWeightTile(wTile, downTileOffset);

                // GEMM: down_partial += inter @ W_down_tile
                // M=tileTokens, N=tileIntermediate(=tileH), K=tileIntermediate
                // isTransB=false (标准矩阵乘)
                AccumulateDownGemm(interLocal, wTile, downOut, iBlock > 0);
            }

            // Cast FP32 → FP16 (就地转换)
            LocalTensor<half> downOutHalf;
            downOutHalf.SetPhyAddr(downOut.GetPhyAddr());
            Cast<half, float>(downOutHalf, downOut, RoundMode::CAST_NONE,
                              actualRows * tileIntermediate_);

            // 写入 workspace (覆写 intermediate 数据)
            DataCopy(workspaceGm_[wsOffset +
                     static_cast<uint64_t>(hBlock) * tileIntermediate_],
                     downOutHalf, actualRows * tileIntermediate_);
        }
    }

    // ====================================================================
    // 辅助函数
    // ====================================================================

    // ------------------------------------------------------------------
    // 加载权重 tile (Gate/Up 投影用, isTransB=true)
    // W[iBlock*tileIntermediate:(iBlock+1)*tileIntermediate, :] shape [NI, H]
    // 需要取第 kBlock 个 K 块列: W[:, kBlock*tileK:(kBlock+1)*tileK]
    // MatmulImpl B 排布 [N, K] = [tileIntermediate, tileK]
    // 从 GM 按行读取: 每行 NI=tI 个元素，但行间距是 hiddenSize
    // ------------------------------------------------------------------
    __aicore__ inline void LoadWeightTile(LocalTensor<half>& wTile,
                                           uint64_t baseOffset,
                                           uint32_t kBlock,
                                           bool isTransB)
    {
        // W_tile 在 GM 中的位置:
        // 行 i (0 <= i < tileIntermediate):
        //   GM addr = baseOffset + i * hiddenSize + kBlock * tileK
        // 连续读取 tileK 个 half 元素
        // 共 tileIntermediate 行
        const uint64_t colOffset = static_cast<uint64_t>(kBlock) * tileK_;

        if (isTransB) {
            // B 需要 [N, K] 排布: 行优先, N=tileIntermediate, K=tileK
            // 行 n: GM[n * hiddenSize + kBlock * tileK] 读 tileK 个元素
            for (uint32_t n = 0; n < tileIntermediate_; n++) {
                DataCopy(wTile[n * tileK_],
                         expertWeightsGm_[baseOffset +
                         static_cast<uint64_t>(n) * hiddenSize_ + colOffset],
                         tileK_);
            }
        } else {
            // B 需要 [K, N] 排布 (标准 row-major)
            // 行 k: GM[baseOffset + k * hiddenSize + colOffset] 读 tileIntermediate 个元素
            // 但此处 hiddenSize 是原始行宽，不适用于 down 投影
            // down 投影的 B 是 [tileIntermediate, tileH], 行宽 = hiddenSize
            for (uint32_t k = 0; k < tileIntermediate_; k++) {
                DataCopy(wTile[k * tileIntermediate_],
                         expertWeightsGm_[baseOffset +
                         static_cast<uint64_t>(k) * hiddenSize_],
                         tileIntermediate_);
            }
        }
    }

    // ------------------------------------------------------------------
    // 加载 Down 投影权重 tile (isTransB=false)
    // W_down[iBlock*tileIntermediate:(iBlock+1)*tileIntermediate,
    //         hBlock*tileH:(hBlock+1)*tileH]
    // 排布 [tileIntermediate, tileH], 行宽 = hiddenSize
    // B 需要 [K, N] = [tileIntermediate, tileH] 排布
    // ------------------------------------------------------------------
    __aicore__ inline void LoadDownWeightTile(LocalTensor<half>& wTile,
                                               uint64_t baseOffset)
    {
        // 行 k (0 <= k < tileIntermediate):
        //   GM = baseOffset + k * hiddenSize, 读 tileIntermediate(=tileH) 个元素
        for (uint32_t k = 0; k < tileIntermediate_; k++) {
            DataCopy(wTile[k * tileIntermediate_],
                     expertWeightsGm_[baseOffset +
                     static_cast<uint64_t>(k) * hiddenSize_],
                     tileIntermediate_);
        }
    }

    // ------------------------------------------------------------------
    // 累加 GEMM: C += A @ B (投影用, isTransB=true)
    // 如果 doAdd == true, 结果加到已有 C 上; 否则覆盖
    // ------------------------------------------------------------------
    __aicore__ inline void AccumulateGemm(const LocalTensor<half>& aLocal,
                                           const LocalTensor<half>& bLocal,
                                           LocalTensor<float>& cLocal,
                                           bool doAdd)
    {
        LocalTensor<float> tempC = tempBuf_.Get<float>();

        mmProj_.SetTensorA(aLocal);
        mmProj_.SetTensorB(bLocal);
        mmProj_.Iterate();
        mmProj_.GetTensorC(tempC);

        if (doAdd) {
            Add<float>(cLocal, cLocal, tempC, tileTokens_ * tileIntermediate_);
        } else {
            // 首次: 直接复制 (Duplicate 已在调用前完成，这里用 Add 等价于赋值)
            DataCopy(cLocal, tempC, tileTokens_ * tileIntermediate_ * sizeof(float));
        }
    }

    // ------------------------------------------------------------------
    // 累加 GEMM: C += A @ B (Down 投影用, isTransB=false)
    // ------------------------------------------------------------------
    __aicore__ inline void AccumulateDownGemm(const LocalTensor<half>& aLocal,
                                               const LocalTensor<half>& bLocal,
                                               LocalTensor<float>& cLocal,
                                               bool doAdd)
    {
        LocalTensor<half> interLocal = interBuf_.Get<half>();
        // 用 interBuf 暂存 MatmulImpl 输出 (Cast 为 float 指针)
        // 注意: interBuf 此时已被 intermediate 数据占用!
        // 改用 projBuf 尾部区域作为 tempC —— 但 projBuf 已被 downOut 占用
        // 解决: 使用独立的临时缓冲策略
        // 实际: MatmulImpl GetTensorC 可以直接输出到 cLocal，然后用 Add 累加
        // 但 GetTensorC 会覆盖目标，不能直接用于累加

        // 方案: 先 GetTensorC 到 cLocal 的一半区域，然后 Add
        // 简化方案: 使用 interBuf 作为临时 (此时 interBuf 在本函数内已不再需要)
        // 不对，interBuf 在 ComputeDownProj 中被循环使用

        // 最佳方案: 直接在 cLocal 上操作
        // 如果 doAdd: 先 GetTensorC 到临时位置，再 Add
        // 如果不 doAdd: 直接 GetTensorC 到 cLocal

        if (doAdd) {
            // 用 xBuf 的前 tileTokens*tileIntermediate*sizeof(float) 字节作为临时
            // xBuf 在 Stage B 中不再用于 x 数据
            LocalTensor<float> tempC;
            tempC.SetPhyAddr(xBuf_.Get<half>().GetPhyAddr());
            mmDown_.SetTensorA(aLocal);
            mmDown_.SetTensorB(bLocal);
            mmDown_.Iterate();
            mmDown_.GetTensorC(tempC);
            Add<float>(cLocal, cLocal, tempC, tileTokens_ * tileIntermediate_);
        } else {
            mmDown_.SetTensorA(aLocal);
            mmDown_.SetTensorB(bLocal);
            mmDown_.Iterate();
            mmDown_.GetTensorC(cLocal);
        }
    }

    // ------------------------------------------------------------------
    // SiLU 激活: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    // 输入: gate [tileTokens * tileIntermediate] FP32
    // 输出: inter [tileTokens * tileIntermediate] FP16
    // ------------------------------------------------------------------
    __aicore__ inline void ComputeSiLU(const LocalTensor<float>& gate,
                                        const LocalTensor<half>& inter,
                                        const LocalTensor<float>& temp)
    {
        const uint32_t n = tileTokens_ * tileIntermediate_;

        // 1. neg_x = -x
        Muls<float>(temp, gate, -1.0f, n);
        // 2. exp(-x)
        Exp<float>(temp, temp, n);
        // 3. 1 + exp(-x)
        Adds<float>(temp, temp, 1.0f, n);
        // 4. sigmoid(x) = 1 / (1 + exp(-x))
        Reciprocal<float>(temp, temp, n);
        // 5. SiLU(x) = x * sigmoid(x)
        Mul<float>(gate, gate, temp, n);
        // 6. Cast FP32 → FP16
        Cast<half, float>(inter, gate, RoundMode::CAST_NONE, n);
    }

    // ------------------------------------------------------------------
    // SwiGLU 逐元素乘: hidden = SiLU(gate) * up → FP16
    // 输入: inter [tileTokens*tileIntermediate] FP16 (已有 SiLU(gate))
    //        up   [tileTokens*tileIntermediate] FP32
    // 输出: inter 被覆写为 FP16 结果
    // ------------------------------------------------------------------
    __aicore__ inline void MulSiLU(const LocalTensor<half>& inter,
                                    const LocalTensor<float>& up,
                                    const LocalTensor<float>& temp)
    {
        const uint32_t n = tileTokens_ * tileIntermediate_;

        // up FP32 → FP16 (用 temp 的前 n*sizeof(half) 字节)
        LocalTensor<half> upHalf;
        upHalf.SetPhyAddr(temp.GetPhyAddr());
        Cast<half, float>(upHalf, up, RoundMode::CAST_NONE, n);

        // inter *= up_half (FP16 逐元素乘)
        Mul<half>(inter, inter, upHalf, n);
    }

private:
    // ---- Tiling 参数 ----
    uint32_t numTokens_;
    uint32_t hiddenSize_;
    uint32_t intermediateSize_;
    uint32_t numExperts_;
    uint32_t topK_;
    uint32_t tileTokens_;
    uint32_t tileIntermediate_;
    uint32_t tileK_;

    // ---- 内存管理 ----
    TPipe pipe_;
    TBuf xBuf_;          // [tileTokens, hiddenSize] FP16 = 64KB
    TBuf wTileBuf_;      // [tileK, tileIntermediate] FP16 = 64KB
    TBuf projBuf_;      // [tileTokens, tileIntermediate] FP32 = 16KB
    TBuf interBuf_;      // [tileTokens, tileIntermediate] FP16 = 8KB
    TBuf tempBuf_;       // [tileTokens, tileIntermediate] FP32 = 16KB

    // MatmulImpl 对象
    MatmulImpl<ProjAType, ProjBType, ProjCType, ProjCType> mmProj_;
    MatmulImpl<DownAType, DownBType, DownCType, DownCType> mmDown_;

    // TCubeTiling 缓冲区
    TBuf projTilingBuf_;
    TBuf downTilingBuf_;

    // ---- 全局内存 ----
    GlobalTensor<half> xGm_;
    GlobalTensor<int32_t> expertIdsGm_;
    GlobalTensor<half> weightsGm_;
    GlobalTensor<half> expertWeightsGm_;
    GlobalTensor<half> expertOutGm_;
    GlobalTensor<half> workspaceGm_;
};

// ============================================================================
// 算子核函数入口
// ============================================================================
extern "C" __global__ __aicore__ void MoeExpertFFN(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR weights,
    GM_ADDR expertWeights, GM_ADDR expertOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KernelMoeExpertFFN op;
    op.Init(x, expertIds, weights, expertWeights, expertOut, workspace, tiling);
    op.Process();
}
