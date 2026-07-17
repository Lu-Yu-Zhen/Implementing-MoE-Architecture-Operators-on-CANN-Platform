/**
 * @file moe_cpu_sim.cpp
 * @brief MoE (Mixture of Experts) 算子 CPU 模拟测试
 *
 * 本文件不依赖 CANN 运行环境，可在任意 x86 机器上编译运行，
 * 用于验证：
 *   1. MoE 三阶段流水线算法正确性 (Gating→TopK→ExpertFFN→Combine)
 *   2. Tiling 参数合理性
 *   3. 分块计算结果与 naive 参考实现是否一致
 *
 * 编译（无需 CANN）：
 *   g++ -O2 -std=c++17 moe_cpu_sim.cpp -o moe_cpu_sim -lm
 *
 * 运行：
 *   ./moe_cpu_sim
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <utility>

// ============================================================================
// 常量与结构体
// ============================================================================

// AI Core 参数（模拟 910B）
static constexpr uint32_t UB_SIZE_BYTES = 256 * 1024;  // 256 KB UB (910B)
static constexpr uint32_t DEFAULT_BLOCK_DIM = 24;       // 910B 共 24 个 AI Core

// MoeRoutingTopK Tiling 参数
struct MoeRoutingTilingData {
    uint32_t numTokens, hiddenSize, numExperts, topK;
    uint32_t tileS;
    uint32_t blockDim, totalTaskNum, coreTaskNum;
};

// MoeExpertFFN Tiling 参数
struct MoeExpertFFNTilingData {
    uint32_t numTokens, hiddenSize, intermediateSize, numExperts, topK;
    uint32_t tileTokens, tileIntermediate, tileK;
    uint32_t blockDim, totalTaskNum;
};

// MoeCombine Tiling 参数
struct MoeCombineTilingData {
    uint32_t numTokens, hiddenSize, topK;
    uint32_t tileS;
    uint32_t blockDim, totalTaskNum;
};

// ============================================================================
// Tiling 计算函数（模拟 host 侧 tiling）
// ============================================================================

// Kernel 1: MoeRoutingTopK Tiling
// W_gate 常驻 UB: hiddenSize * numExperts * sizeof(float)
// x_tile: tileS * hiddenSize * sizeof(float)
// logits: tileS * numExperts * sizeof(float) (FP32 累加)
// UB 约束: W_gate_bytes + tileS*H*4 + tileS*E*4 <= UB_SIZE
MoeRoutingTilingData ComputeRoutingTiling(uint32_t numTokens, uint32_t hiddenSize,
                                           uint32_t numExperts, uint32_t topK)
{
    MoeRoutingTilingData t{};
    t.numTokens = numTokens;
    t.hiddenSize = hiddenSize;
    t.numExperts = numExperts;
    t.topK = topK;
    t.blockDim = DEFAULT_BLOCK_DIM;

    // W_gate 常驻（FP32 仿真）: H * E * 4 bytes
    uint32_t wGateBytes = hiddenSize * numExperts * 4;
    uint32_t remainUB = UB_SIZE_BYTES - wGateBytes;
    uint32_t bytesPerToken = hiddenSize * 4 + numExperts * 4;
    uint32_t tileS = remainUB / bytesPerToken;
    // 对齐到 16 的倍数，上限 32
    tileS = (tileS / 16) * 16;
    if (tileS == 0) tileS = 1;
    if (tileS > 32) tileS = 32;
    if (tileS > numTokens) tileS = numTokens;

    t.tileS = tileS;
    t.totalTaskNum = (numTokens + tileS - 1) / tileS;
    t.coreTaskNum = (t.totalTaskNum + t.blockDim - 1) / t.blockDim;
    return t;
}

// Kernel 2: MoeExpertFFN Tiling
// 两阶段设计 (tileTokens=16)，UB 取 max(Stage_A, Stage_B):
// Stage A (SwiGLU): xBuf + wTileBuf + projBuf + interBuf + tempBuf ≈ 170KB
// Stage B (Down):   interBuf + wTileBuf + projBuf + tempBuf ≈ 104KB
MoeExpertFFNTilingData ComputeExpertFFNTiling(uint32_t numTokens, uint32_t hiddenSize,
                                               uint32_t intermediateSize,
                                               uint32_t numExperts, uint32_t topK)
{
    MoeExpertFFNTilingData t{};
    t.numTokens = numTokens;
    t.hiddenSize = hiddenSize;
    t.intermediateSize = intermediateSize;
    t.numExperts = numExperts;
    t.topK = topK;
    t.blockDim = DEFAULT_BLOCK_DIM;

    // 默认分块参数 (tileTokens=16，与 kernel UB 预算一致)
    t.tileTokens = 16;
    t.tileIntermediate = 256;
    t.tileK = 128;

    if (t.tileTokens > numTokens) t.tileTokens = numTokens;
    if (t.tileIntermediate > intermediateSize) t.tileIntermediate = intermediateSize;
    if (t.tileK > hiddenSize) t.tileK = hiddenSize;

    // 任务分配: 每专家一个任务
    t.totalTaskNum = numExperts;
    return t;
}

// Kernel 3: MoeCombine Tiling
// expert_out_tile: tileS * topK * hiddenSize * 4 bytes
// y_tile: tileS * hiddenSize * 4 bytes
// UB 约束: tileS * (topK + 1) * hiddenSize * 4 <= UB_SIZE
MoeCombineTilingData ComputeCombineTiling(uint32_t numTokens, uint32_t hiddenSize,
                                           uint32_t topK)
{
    MoeCombineTilingData t{};
    t.numTokens = numTokens;
    t.hiddenSize = hiddenSize;
    t.topK = topK;
    t.blockDim = DEFAULT_BLOCK_DIM;

    uint32_t bytesPerToken = (topK + 1) * hiddenSize * 4;
    uint32_t tileS = UB_SIZE_BYTES / bytesPerToken;
    tileS = (tileS / 16) * 16;
    if (tileS == 0) tileS = 1;
    if (tileS > 64) tileS = 64;
    if (tileS > numTokens) tileS = numTokens;

    t.tileS = tileS;
    t.totalTaskNum = (numTokens + tileS - 1) / tileS;
    return t;
}

// ============================================================================
// 辅助函数
// ============================================================================

// SiLU 激活: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
static inline float silu(float x)
{
    return x / (1.0f + expf(-x));
}

// 生成正态分布随机数据，均值 0，方差 1/sqrt(dim)
void GenNormalData(std::vector<float>& data, size_t n, int dim, std::mt19937& rng)
{
    std::normal_distribution<float> dist(0.0f, 1.0f / sqrtf((float)dim));
    data.resize(n);
    for (size_t i = 0; i < n; i++)
        data[i] = dist(rng);
}

// ============================================================================
// Naive MoE 参考实现（完整流水线，不分块）
// ----------------------------------------------------------------------------
// 输入:
//   x:              [numTokens, hiddenSize]          - 输入 token 矩阵
//   w_gate:         [hiddenSize, numExperts]         - 门控权重矩阵
//   expert_weights: [numExperts * 3 * I * H]         - 所有专家的 FFN 权重
//     排布: expert_weights[e * 3 * I * H + offset]
//       W_gate_e: offset = 0*I*H,           shape [I, H] (gate_proj 权重)
//       W_up_e:   offset = 1*I*H,           shape [I, H] (up_proj 权重)
//       W_down_e: offset = 2*I*H,           shape [H, I] (down_proj 权重)
// 输出:
//   output:         [numTokens, hiddenSize]          - MoE 输出
// ============================================================================
void NaiveMoE(const std::vector<float>& x,
              const std::vector<float>& w_gate,
              const std::vector<float>& expert_weights,
              int numTokens, int hiddenSize, int numExperts,
              int topK, int intermediateSize,
              std::vector<float>& output)
{
    const int H = hiddenSize;
    const int E = numExperts;
    const int K = topK;
    const int I = intermediateSize;
    const int T = numTokens;

    output.assign((size_t)T * H, 0.0f);

    // ---- Step 1: Gating ---- logits = x @ w_gate → [T, E]
    std::vector<float> logits((size_t)T * E, 0.0f);
    for (int t = 0; t < T; t++) {
        for (int e = 0; e < E; e++) {
            float dot = 0.0f;
            for (int h = 0; h < H; h++)
                dot += x[t * H + h] * w_gate[h * E + e];
            logits[t * E + e] = dot;
        }
    }

    // ---- Step 2: Softmax ---- 对每行 logits 做 softmax 归一化
    std::vector<float> probs((size_t)T * E);
    for (int t = 0; t < T; t++) {
        float maxVal = -1e30f;
        for (int e = 0; e < E; e++)
            if (logits[t * E + e] > maxVal) maxVal = logits[t * E + e];
        float sumExp = 0.0f;
        for (int e = 0; e < E; e++) {
            probs[t * E + e] = expf(logits[t * E + e] - maxVal);
            sumExp += probs[t * E + e];
        }
        for (int e = 0; e < E; e++)
            probs[t * E + e] /= sumExp;
    }

    // ---- Step 3: TopK ---- 对每行选 Top-K 专家
    std::vector<int>   expert_ids((size_t)T * K);
    std::vector<float> weights((size_t)T * K);
    for (int t = 0; t < T; t++) {
        // 用简单选择法找 top-K
        std::vector<int> indices(E);
        std::iota(indices.begin(), indices.end(), 0);
        // 部分排序：前 K 个最大
        std::partial_sort(indices.begin(), indices.begin() + K, indices.end(),
            [&](int a, int b) { return probs[t * E + a] > probs[t * E + b]; });

        // 对选中的 K 个专家权重做归一化
        float wSum = 0.0f;
        for (int k = 0; k < K; k++) {
            expert_ids[t * K + k] = indices[k];
            wSum += probs[t * E + indices[k]];
        }
        for (int k = 0; k < K; k++)
            weights[t * K + k] = probs[t * E + indices[k]] / wSum;
    }

    // ---- Step 4: Expert FFN ---- 对每个 token 的每个选中专家执行 FFN
    // ffn_out[token][k] 存储第 k 个专家的输出 [hiddenSize]
    std::vector<float> ffn_out((size_t)T * K * H, 0.0f);

    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            int e = expert_ids[t * K + k];

            // 获取专家权重指针
            // W_gate_e: [I, H] - 排布为行优先，第 i 行第 h 列 = e*3*I*H + 0*I*H + i*H + h
            // W_up_e:   [I, H] - 排布为行优先，第 i 行第 h 列 = e*3*I*H + 1*I*H + i*H + h
            // W_down_e: [H, I] - 排布为行优先，第 h 行第 i 列 = e*3*I*H + 2*I*H + h*I + i
            size_t expertBase = (size_t)e * 3 * I * H;
            const float* W_gate_e = &expert_weights[expertBase + 0];
            const float* W_up_e   = &expert_weights[expertBase + (size_t)I * H];
            const float* W_down_e = &expert_weights[expertBase + (size_t)2 * I * H];

            // gate_proj = x_token @ W_gate_e^T → [I]
            // gate_proj[i] = sum_h x[h] * W_gate_e[i, h]
            std::vector<float> gate_proj(I);
            for (int i = 0; i < I; i++) {
                float dot = 0.0f;
                for (int h = 0; h < H; h++)
                    dot += x[t * H + h] * W_gate_e[i * H + h];
                gate_proj[i] = dot;
            }

            // up_proj = x_token @ W_up_e^T → [I]
            // up_proj[i] = sum_h x[h] * W_up_e[i, h]
            std::vector<float> up_proj(I);
            for (int i = 0; i < I; i++) {
                float dot = 0.0f;
                for (int h = 0; h < H; h++)
                    dot += x[t * H + h] * W_up_e[i * H + h];
                up_proj[i] = dot;
            }

            // hidden = SiLU(gate_proj) * up_proj → [I]  (SwiGLU)
            std::vector<float> hidden(I);
            for (int i = 0; i < I; i++)
                hidden[i] = silu(gate_proj[i]) * up_proj[i];

            // ffn_out = hidden @ W_down_e^T → [H]
            // ffn_out[h] = sum_i hidden[i] * W_down_e[h, i]
            for (int h = 0; h < H; h++) {
                float dot = 0.0f;
                for (int i = 0; i < I; i++)
                    dot += hidden[i] * W_down_e[h * I + i];
                ffn_out[t * K * H + k * H + h] = dot;
            }
        }
    }

    // ---- Step 5: Combine ---- output[token] = Σ weight_k * ffn_out_k
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            float w = weights[t * K + k];
            for (int h = 0; h < H; h++)
                output[t * H + h] += w * ffn_out[t * K * H + k * H + h];
        }
    }
}

// ============================================================================
// Tiled MoE 模拟实现（模拟 3-Kernel 分块计算）
// ============================================================================

// ----------------------------------------------------------------------------
// TiledRouting: 按 tileS 分块计算 logits、softmax、TopK
// ----------------------------------------------------------------------------
void TiledRouting(const std::vector<float>& x,
                  const std::vector<float>& w_gate,
                  int numTokens, int hiddenSize, int numExperts, int topK,
                  const MoeRoutingTilingData& tiling,
                  std::vector<int>& expert_ids,
                  std::vector<float>& weights)
{
    const int H = hiddenSize, E = numExperts, K = topK, T = numTokens;
    const uint32_t tileS = tiling.tileS;

    expert_ids.assign((size_t)T * K, 0);
    weights.assign((size_t)T * K, 0.0f);

    // 模拟 UB 缓冲区
    std::vector<float> xTile(tileS * H);       // x 分块
    std::vector<float> wGateBuf(H * E);        // W_gate 常驻
    std::vector<float> logitsTile(tileS * E);  // logits 分块

    // W_gate 常驻 UB
    for (int i = 0; i < H * E; i++)
        wGateBuf[i] = w_gate[i];

    uint32_t numTiles = (T + tileS - 1) / tileS;
    for (uint32_t tile = 0; tile < numTiles; tile++) {
        uint32_t startToken = tile * tileS;
        uint32_t actualTokens = (startToken + tileS <= (uint32_t)T) ? tileS : ((uint32_t)T - startToken);

        // 搬入 x 块到 UB
        for (uint32_t i = 0; i < actualTokens; i++)
            for (int h = 0; h < H; h++)
                xTile[i * H + h] = x[(startToken + i) * H + h];

        // 计算 logits = x_tile @ w_gate → [actualTokens, E]
        for (uint32_t i = 0; i < actualTokens; i++) {
            for (int e = 0; e < E; e++) {
                float dot = 0.0f;
                for (int h = 0; h < H; h++)
                    dot += xTile[i * H + h] * wGateBuf[h * E + e];
                logitsTile[i * E + e] = dot;
            }
        }

        // Softmax + TopK（在 tile 内逐 token 处理）
        for (uint32_t i = 0; i < actualTokens; i++) {
            int tIdx = startToken + i;

            // Softmax
            float maxVal = -1e30f;
            for (int e = 0; e < E; e++)
                if (logitsTile[i * E + e] > maxVal) maxVal = logitsTile[i * E + e];
            float sumExp = 0.0f;
            std::vector<float> probs(E);
            for (int e = 0; e < E; e++) {
                probs[e] = expf(logitsTile[i * E + e] - maxVal);
                sumExp += probs[e];
            }
            for (int e = 0; e < E; e++)
                probs[e] /= sumExp;

            // TopK 选择
            std::vector<int> indices(E);
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + K, indices.end(),
                [&](int a, int b) { return probs[a] > probs[b]; });

            float wSum = 0.0f;
            for (int k = 0; k < K; k++) {
                expert_ids[tIdx * K + k] = indices[k];
                wSum += probs[indices[k]];
            }
            for (int k = 0; k < K; k++)
                weights[tIdx * K + k] = probs[indices[k]] / wSum;
        }
    }
}

// ----------------------------------------------------------------------------
// TiledExpertFFN: 按专家遍历，每个专家内按 tileTokens 分块，
//                 FFN 内按 tileIntermediate 分块 GEMM
// ----------------------------------------------------------------------------
void TiledExpertFFN(const std::vector<float>& x,
                    const std::vector<float>& expert_weights,
                    const std::vector<int>& expert_ids,
                    const std::vector<float>& route_weights,
                    int numTokens, int hiddenSize, int numExperts,
                    int topK, int intermediateSize,
                    const MoeExpertFFNTilingData& tiling,
                    std::vector<float>& ffn_out)
{
    const int H = hiddenSize, E = numExperts, K = topK, I = intermediateSize, T = numTokens;
    const uint32_t tileTokens = tiling.tileTokens;
    const uint32_t tileInter = tiling.tileIntermediate;

    // ffn_out: [T, K, H]
    ffn_out.assign((size_t)T * K * H, 0.0f);

    // 对每个专家，收集分配给它的 token，分块处理
    for (int e = 0; e < E; e++) {
        // 收集被分配到该专家的 (token, k_slot) 对
        std::vector<std::pair<int,int>> assigned; // (tokenIdx, kSlot)
        for (int t = 0; t < T; t++)
            for (int k = 0; k < K; k++)
                if (expert_ids[t * K + k] == e)
                    assigned.push_back({t, k});

        if (assigned.empty()) continue;

        // 专家权重基地址
        size_t expertBase = (size_t)e * 3 * I * H;
        const float* W_gate_e = &expert_weights[expertBase + 0];
        const float* W_up_e   = &expert_weights[expertBase + (size_t)I * H];
        const float* W_down_e = &expert_weights[expertBase + (size_t)2 * I * H];

        // 按 tileTokens 分块处理
        uint32_t numAssigned = (uint32_t)assigned.size();
        uint32_t numTokenTiles = (numAssigned + tileTokens - 1) / tileTokens;

        for (uint32_t tt = 0; tt < numTokenTiles; tt++) {
            uint32_t tokStart = tt * tileTokens;
            uint32_t actualTok = (tokStart + tileTokens <= numAssigned) ?
                                  tileTokens : (numAssigned - tokStart);

            // 对当前 token 块中的每个 token 做 FFN
            for (uint32_t ti = 0; ti < actualTok; ti++) {
                int tokenIdx = assigned[tokStart + ti].first;
                int kSlot    = assigned[tokStart + ti].second;

                // 按 tileIntermediate 分块计算 gate_proj, up_proj
                std::vector<float> gate_proj(I, 0.0f);
                std::vector<float> up_proj(I, 0.0f);

                uint32_t numInterTiles = (I + tileInter - 1) / tileInter;
                for (uint32_t it = 0; it < numInterTiles; it++) {
                    uint32_t iStart = it * tileInter;
                    uint32_t actualI = (iStart + tileInter <= (uint32_t)I) ?
                                        tileInter : ((uint32_t)I - iStart);

                    // 计算 gate_proj[iStart..iStart+actualI) 和 up_proj[...]
                    for (uint32_t ii = 0; ii < actualI; ii++) {
                        int i = iStart + ii;
                        float gDot = 0.0f, uDot = 0.0f;
                        for (int h = 0; h < H; h++) {
                            float xVal = x[tokenIdx * H + h];
                            gDot += xVal * W_gate_e[i * H + h];
                            uDot += xVal * W_up_e[i * H + h];
                        }
                        gate_proj[i] = gDot;
                        up_proj[i] = uDot;
                    }
                }

                // SwiGLU: hidden = SiLU(gate_proj) * up_proj
                std::vector<float> hidden(I);
                for (int i = 0; i < I; i++)
                    hidden[i] = silu(gate_proj[i]) * up_proj[i];

                // down_proj: ffn_out[h] = sum_i hidden[i] * W_down_e[h, i]
                // 按 tileIntermediate 分块累加
                std::vector<float> outAccum(H, 0.0f);
                for (uint32_t it = 0; it < numInterTiles; it++) {
                    uint32_t iStart = it * tileInter;
                    uint32_t actualI = (iStart + tileInter <= (uint32_t)I) ?
                                        tileInter : ((uint32_t)I - iStart);
                    for (int h = 0; h < H; h++) {
                        float dot = 0.0f;
                        for (uint32_t ii = 0; ii < actualI; ii++) {
                            int i = iStart + ii;
                            dot += hidden[i] * W_down_e[h * I + i];
                        }
                        outAccum[h] += dot;
                    }
                }

                // 写入 ffn_out
                for (int h = 0; h < H; h++)
                    ffn_out[tokenIdx * K * H + kSlot * H + h] = outAccum[h];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// TiledCombine: 按 tileS 分块做加权合并
// ----------------------------------------------------------------------------
void TiledCombine(const std::vector<float>& ffn_out,
                  const std::vector<int>& expert_ids,
                  const std::vector<float>& route_weights,
                  int numTokens, int hiddenSize, int topK,
                  const MoeCombineTilingData& tiling,
                  std::vector<float>& output)
{
    const int H = hiddenSize, K = topK, T = numTokens;
    const uint32_t tileS = tiling.tileS;

    output.assign((size_t)T * H, 0.0f);

    // 模拟 UB 缓冲区
    std::vector<float> outTile(tileS * H, 0.0f);

    uint32_t numTiles = (T + tileS - 1) / tileS;
    for (uint32_t tile = 0; tile < numTiles; tile++) {
        uint32_t startToken = tile * tileS;
        uint32_t actualTokens = (startToken + tileS <= (uint32_t)T) ?
                                 tileS : ((uint32_t)T - startToken);

        // 清零 tile 缓冲区
        std::fill(outTile.begin(), outTile.begin() + actualTokens * H, 0.0f);

        // 加权合并: y[t] = Σ_k weight[k] * ffn_out[t][k]
        for (uint32_t i = 0; i < actualTokens; i++) {
            int tIdx = startToken + i;
            for (int k = 0; k < K; k++) {
                float w = route_weights[tIdx * K + k];
                for (int h = 0; h < H; h++)
                    outTile[i * H + h] += w * ffn_out[tIdx * K * H + k * H + h];
            }
        }

        // 写回全局输出
        for (uint32_t i = 0; i < actualTokens; i++)
            for (int h = 0; h < H; h++)
                output[(startToken + i) * H + h] = outTile[i * H + h];
    }
}

// ----------------------------------------------------------------------------
// TiledMoE: 完整的 3-Kernel 分块模拟
// ----------------------------------------------------------------------------
void TiledMoE(const std::vector<float>& x,
              const std::vector<float>& w_gate,
              const std::vector<float>& expert_weights,
              int numTokens, int hiddenSize, int numExperts,
              int topK, int intermediateSize,
              std::vector<float>& output)
{
    // Kernel 1: Routing
    auto routingTiling = ComputeRoutingTiling(numTokens, hiddenSize, numExperts, topK);
    std::vector<int> expert_ids;
    std::vector<float> route_weights;
    TiledRouting(x, w_gate, numTokens, hiddenSize, numExperts, topK,
                 routingTiling, expert_ids, route_weights);

    // Kernel 2: Expert FFN
    auto ffnTiling = ComputeExpertFFNTiling(numTokens, hiddenSize, intermediateSize,
                                             numExperts, topK);
    std::vector<float> ffn_out;
    TiledExpertFFN(x, expert_weights, expert_ids, route_weights,
                   numTokens, hiddenSize, numExperts, topK, intermediateSize,
                   ffnTiling, ffn_out);

    // Kernel 3: Combine
    auto combineTiling = ComputeCombineTiling(numTokens, hiddenSize, topK);
    TiledCombine(ffn_out, expert_ids, route_weights,
                 numTokens, hiddenSize, topK, combineTiling, output);
}

// ============================================================================
// 测试框架
// ============================================================================

struct TestResult {
    std::string name;
    bool passed;
    float maxAbsErr;
    float meanAbsErr;
    double cosSim;
    double naiveMs;
    double tiledMs;
};

// 比较两个向量: 最大/平均绝对误差 + 余弦相似度
void CompareResults(const std::vector<float>& actual,
                    const std::vector<float>& expected,
                    float& maxAbsErr, float& meanAbsErr, double& cosSim)
{
    maxAbsErr = 0.0f;
    float sumAbsErr = 0.0f;
    double dotAB = 0.0, dotAA = 0.0, dotBB = 0.0;

    for (size_t i = 0; i < actual.size(); i++) {
        float err = fabsf(actual[i] - expected[i]);
        sumAbsErr += err;
        if (err > maxAbsErr) maxAbsErr = err;
        dotAB += (double)actual[i] * (double)expected[i];
        dotAA += (double)actual[i] * (double)actual[i];
        dotBB += (double)expected[i] * (double)expected[i];
    }
    meanAbsErr = sumAbsErr / actual.size();
    cosSim = (dotAA > 0 && dotBB > 0) ? dotAB / (sqrt(dotAA) * sqrt(dotBB)) : 0.0;
}

// ============================================================================
// 测试用例
// ============================================================================
TestResult RunTest(const char* name, int numTokens, int hiddenSize,
                   int numExperts, int topK, int intermediateSize,
                   uint32_t seed = 42, bool timing = false)
{
    printf("\n--- %s ---\n", name);
    printf("  T=%d H=%d E=%d K=%d I=%d\n",
           numTokens, hiddenSize, numExperts, topK, intermediateSize);

    std::mt19937 rng(seed);

    // 生成随机输入
    std::vector<float> x, w_gate, expert_weights;
    GenNormalData(x, (size_t)numTokens * hiddenSize, hiddenSize, rng);
    GenNormalData(w_gate, (size_t)hiddenSize * numExperts, hiddenSize, rng);
    // expert_weights: [numExperts * 3 * I * H]
    GenNormalData(expert_weights, (size_t)numExperts * 3 * intermediateSize * hiddenSize,
                  hiddenSize, rng);

    // Naive 参考实现
    std::vector<float> output_naive, output_tiled;
    auto t0 = std::chrono::high_resolution_clock::now();
    NaiveMoE(x, w_gate, expert_weights, numTokens, hiddenSize, numExperts,
             topK, intermediateSize, output_naive);
    auto t1 = std::chrono::high_resolution_clock::now();
    double naiveMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Tiled 分块模拟
    t0 = std::chrono::high_resolution_clock::now();
    TiledMoE(x, w_gate, expert_weights, numTokens, hiddenSize, numExperts,
             topK, intermediateSize, output_tiled);
    t1 = std::chrono::high_resolution_clock::now();
    double tiledMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 打印 Tiling 参数
    auto rt = ComputeRoutingTiling(numTokens, hiddenSize, numExperts, topK);
    auto ft = ComputeExpertFFNTiling(numTokens, hiddenSize, intermediateSize, numExperts, topK);
    auto ct = ComputeCombineTiling(numTokens, hiddenSize, topK);
    printf("  Routing  Tiling: tileS=%u tasks=%u coreTasks=%u\n",
           rt.tileS, rt.totalTaskNum, rt.coreTaskNum);
    printf("  ExpertFFN Tiling: tileTokens=%u tileInter=%u tileK=%u tasks=%u\n",
           ft.tileTokens, ft.tileIntermediate, ft.tileK, ft.totalTaskNum);
    printf("  Combine  Tiling: tileS=%u tasks=%u\n",
           ct.tileS, ct.totalTaskNum);

    if (timing) {
        printf("  Naive:  %.2f ms\n", naiveMs);
        printf("  Tiled:  %.2f ms\n", tiledMs);
    }

    // 精度比较
    float maxAbsErr, meanAbsErr;
    double cosSim;
    CompareResults(output_tiled, output_naive, maxAbsErr, meanAbsErr, cosSim);
    printf("  Max Abs Err:     %.8f\n", maxAbsErr);
    printf("  Mean Abs Err:    %.8f\n", meanAbsErr);
    printf("  Cosine Similarity: %.10f\n", cosSim);

    bool passed = (maxAbsErr < 0.05f && cosSim > 0.999);
    printf("  Result: %s\n", passed ? "PASS" : "FAIL");

    return {name, passed, maxAbsErr, meanAbsErr, cosSim, naiveMs, tiledMs};
}

// ============================================================================
// Tiling 参数验证测试
// ============================================================================
bool TestTilingParams()
{
    printf("\n=== Tiling 参数验证 ===\n");
    bool allPassed = true;

    // 测试 1: Routing 标准参数
    {
        auto t = ComputeRoutingTiling(128, 2048, 16, 2);
        // W_gate = 2048*16*4 = 128KB, 剩余 128KB
        // bytesPerToken = 2048*4 + 16*4 = 8256, tileS = 128*1024/8256 ≈ 15 → 16
        bool ok = (t.tileS > 0 && t.tileS <= 32 && t.totalTaskNum > 0);
        printf("  [Routing] H=2048 E=16 tileS=%u tasks=%u -> %s\n",
               t.tileS, t.totalTaskNum, ok ? "PASS" : "FAIL");
        allPassed &= ok;
    }

    // 测试 2: ExpertFFN 参数
    {
        auto t = ComputeExpertFFNTiling(128, 2048, 8192, 16, 2);
        bool ok = (t.tileTokens == 16 && t.tileIntermediate == 256 &&
                   t.tileK == 128 && t.totalTaskNum == 16);
        printf("  [ExpertFFN] tileTokens=%u tileInter=%u tileK=%u tasks=%u -> %s\n",
               t.tileTokens, t.tileIntermediate, t.tileK, t.totalTaskNum,
               ok ? "PASS" : "FAIL");
        allPassed &= ok;
    }

    // 测试 3: Combine 参数
    {
        auto t = ComputeCombineTiling(128, 2048, 2);
        // bytesPerToken = 3*2048*4 = 24576, tileS = 256*1024/24576 ≈ 10 → 无对齐限制
        bool ok = (t.tileS > 0 && t.tileS <= 64 && t.totalTaskNum > 0);
        printf("  [Combine] H=2048 K=2 tileS=%u tasks=%u -> %s\n",
               t.tileS, t.totalTaskNum, ok ? "PASS" : "FAIL");
        allPassed &= ok;
    }

    // 测试 4: 小规模参数
    {
        auto t = ComputeRoutingTiling(8, 64, 4, 2);
        bool ok = (t.tileS <= 8 && t.totalTaskNum >= 1);
        printf("  [Routing小] T=8 H=64 E=4 tileS=%u tasks=%u -> %s\n",
               t.tileS, t.totalTaskNum, ok ? "PASS" : "FAIL");
        allPassed &= ok;
    }

    // 测试 5: 非对齐 token 数
    {
        auto t = ComputeRoutingTiling(50, 2048, 16, 2);
        bool ok = (t.totalTaskNum == (50 + t.tileS - 1) / t.tileS);
        printf("  [Routing尾块] T=50 tileS=%u tasks=%u -> %s\n",
               t.tileS, t.totalTaskNum, ok ? "PASS" : "FAIL");
        allPassed &= ok;
    }

    return allPassed;
}

// ============================================================================
// UB 容量验证
// ============================================================================
bool TestUBCapacity()
{
    printf("\n=== UB 容量验证 ===\n");
    bool allPassed = true;

    // 验证 1: Routing UB (H=2048, E=16)
    {
        uint32_t H = 2048, E = 16;
        auto t = ComputeRoutingTiling(128, H, E, 2);
        uint32_t wGateBytes = H * E * 4;
        uint32_t xTileBytes = t.tileS * H * 4;
        uint32_t logitsBytes = t.tileS * E * 4;
        uint32_t totalBytes = wGateBytes + xTileBytes + logitsBytes;
        bool ok = (totalBytes <= UB_SIZE_BYTES);
        printf("  [Routing] W_gate=%uKB x_tile=%uKB logits=%uKB total=%uKB (%uKB) -> %s\n",
               wGateBytes/1024, xTileBytes/1024, logitsBytes/1024,
               totalBytes/1024, UB_SIZE_BYTES/1024, ok ? "PASS" : "FAIL");
        allPassed &= ok;
    }

    // 验证 2: ExpertFFN UB (H=2048, I=8192)
    // 两阶段设计，取 max(Stage_A, Stage_B)
    // kernel 使用 FP16 (2 bytes)，CPU 仿真验证时按 kernel 实际数据类型的字节数计算
    {
        uint32_t H = 2048, I = 8192;
        auto t = ComputeExpertFFNTiling(128, H, I, 16, 2);
        // Stage A (SwiGLU): xBuf + wTileBuf + projBuf(FP32) + interBuf(FP16) + tempBuf(FP32)
        uint32_t xBufBytes      = t.tileTokens * H * 2;                    // 16*2048*2 = 64KB (FP16)
        uint32_t wTileBufBytes  = t.tileK * t.tileIntermediate * 2;        // 128*256*2 = 64KB (FP16)
        uint32_t projBufBytes   = t.tileTokens * t.tileIntermediate * 4;   // 16*256*4  = 16KB (FP32)
        uint32_t interBufBytes  = t.tileTokens * t.tileIntermediate * 2;   // 16*256*2  = 8KB  (FP16)
        uint32_t tempBufBytes   = t.tileTokens * t.tileIntermediate * 4;   // 16*256*4  = 16KB (FP32)
        uint32_t stageABytes    = xBufBytes + wTileBufBytes + projBufBytes + interBufBytes + tempBufBytes;
        // Stage B (Down proj): interBuf + wTileBuf + projBuf(FP32) + tempBuf(FP32)
        // interBuf 在 Stage B 中复用为读缓冲，xBuf 复用为 MatmulImpl 临时输出
        uint32_t stageBBytes    = interBufBytes + wTileBufBytes + projBufBytes + tempBufBytes;
        uint32_t totalBytes     = (stageABytes > stageBBytes) ? stageABytes : stageBBytes;
        bool ok = (totalBytes <= UB_SIZE_BYTES);
        printf("  [ExpertFFN] StageA: x=%uKB W=%uKB proj=%uKB inter=%uKB temp=%uKB = %uKB\n",
               xBufBytes/1024, wTileBufBytes/1024, projBufBytes/1024,
               interBufBytes/1024, tempBufBytes/1024, stageABytes/1024);
        printf("             StageB: inter=%uKB W=%uKB proj=%uKB temp=%uKB = %uKB\n",
               interBufBytes/1024, wTileBufBytes/1024, projBufBytes/1024,
               tempBufBytes/1024, stageBBytes/1024);
        printf("             max=%uKB (%uKB) -> %s\n",
               totalBytes/1024, UB_SIZE_BYTES/1024, ok ? "PASS" : "FAIL");
        allPassed &= ok;
    }

    // 验证 3: Combine UB (H=2048, K=2)
    {
        uint32_t H = 2048, K = 2;
        auto t = ComputeCombineTiling(128, H, K);
        uint32_t expertOutBytes = t.tileS * K * H * 4;
        uint32_t yTileBytes = t.tileS * H * 4;
        uint32_t totalBytes = expertOutBytes + yTileBytes;
        bool ok = (totalBytes <= UB_SIZE_BYTES);
        printf("  [Combine] expert_out=%uKB y=%uKB total=%uKB (%uKB) -> %s\n",
               expertOutBytes/1024, yTileBytes/1024,
               totalBytes/1024, UB_SIZE_BYTES/1024, ok ? "PASS" : "FAIL");
        allPassed &= ok;
    }

    // 验证 4: 中规模 (H=256, E=8, I=1024)
    {
        uint32_t H = 256, E = 8, I = 1024;
        auto rt = ComputeRoutingTiling(32, H, E, 2);
        uint32_t rWGate = H * E * 4;
        uint32_t rXTile = rt.tileS * H * 4;
        uint32_t rLogits = rt.tileS * E * 4;
        uint32_t rTotal = rWGate + rXTile + rLogits;
        bool rOk = (rTotal <= UB_SIZE_BYTES);
        printf("  [中规模 Routing] total=%uKB (%uKB) -> %s\n",
               rTotal/1024, UB_SIZE_BYTES/1024, rOk ? "PASS" : "FAIL");
        allPassed &= rOk;

        auto ft = ComputeExpertFFNTiling(32, H, I, E, 2);
        // 两阶段设计，按 kernel 实际 FP16 字节数计算
        uint32_t fXBuf     = ft.tileTokens * H * 2;
        uint32_t fWTile    = ft.tileK * ft.tileIntermediate * 2;
        uint32_t fProjBuf  = ft.tileTokens * ft.tileIntermediate * 4;
        uint32_t fInterBuf = ft.tileTokens * ft.tileIntermediate * 2;
        uint32_t fTempBuf  = ft.tileTokens * ft.tileIntermediate * 4;
        uint32_t fStageA   = fXBuf + fWTile + fProjBuf + fInterBuf + fTempBuf;
        uint32_t fStageB   = fInterBuf + fWTile + fProjBuf + fTempBuf;
        uint32_t fTotal    = (fStageA > fStageB) ? fStageA : fStageB;
        bool fOk = (fTotal <= UB_SIZE_BYTES);
        printf("  [中规模 ExpertFFN] StageA=%uKB StageB=%uKB max=%uKB (%uKB) -> %s\n",
               fStageA/1024, fStageB/1024, fTotal/1024,
               UB_SIZE_BYTES/1024, fOk ? "PASS" : "FAIL");
        allPassed &= fOk;
    }

    return allPassed;
}

// ============================================================================
// Main
// ============================================================================
int main()
{
    printf("========================================\n");
    printf(" MoE CPU 仿真测试\n");
    printf(" (三阶段流水线算法验证)\n");
    printf("========================================\n");
    printf("[INFO] MoE 三阶段流水线:\n");
    printf("  Kernel 1: MoeRoutingTopK  - 门控路由 + Top-K 选择\n");
    printf("  Kernel 2: MoeExpertFFN    - 专家 FFN (GEMM + SwiGLU)\n");
    printf("  Kernel 3: MoeCombine      - 加权合并专家输出\n");
    printf("[INFO] 目标硬件: Ascend 910B (24 AI Cores, ~256KB UB/core)\n");

    // 1. Tiling 参数验证
    bool tilingOk = TestTilingParams();
    printf("\nTiling 参数验证: %s\n", tilingOk ? "ALL PASS" : "FAILED");

    // 2. 精度验证测试用例
    std::vector<TestResult> results;

    // 小规模: T=8, H=64, E=4, K=2, I=256
    results.push_back(RunTest("小规模 T=8 H=64 E=4 K=2 I=256",
                              8, 64, 4, 2, 256, 42));

    // 中规模: T=32, H=256, E=8, K=2, I=1024
    results.push_back(RunTest("中规模 T=32 H=256 E=8 K=2 I=1024",
                              32, 256, 8, 2, 1024, 43));

    // 目标规模: T=128, H=2048, E=16, K=2, I=8192（含计时）
    results.push_back(RunTest("目标规模 T=128 H=2048 E=16 K=2 I=8192",
                              128, 2048, 16, 2, 8192, 44, true));

    // 3. UB 容量验证
    bool ubOk = TestUBCapacity();
    printf("\nUB 容量验证: %s\n", ubOk ? "ALL PASS" : "FAILED");

    // 4. 测试汇总
    printf("\n========================================\n");
    printf(" 测试汇总\n");
    printf("========================================\n");
    printf("%-45s %-6s %-12s %-12s %-14s\n",
           "Test", "Result", "MaxErr", "MeanErr", "CosineSim");
    printf("%-45s %-6s %-12s %-12s %-14s\n",
           "----", "------", "------", "-------", "-----------");

    int passCount = 0;
    for (const auto& r : results) {
        printf("%-45s %-6s %.2e     %.2e     %.10f\n",
               r.name.c_str(),
               r.passed ? "PASS" : "FAIL",
               r.maxAbsErr,
               r.meanAbsErr,
               r.cosSim);
        if (r.passed) passCount++;
    }

    printf("\nTiling 参数: %s\n", tilingOk ? "PASS" : "FAIL");
    printf("UB 容量:    %s\n", ubOk ? "PASS" : "FAIL");
    printf("精度测试:   %d/%d PASS\n", passCount, (int)results.size());

    if (tilingOk && ubOk && passCount == (int)results.size()) {
        printf("\n>>> ALL TESTS PASSED <<<\n");
        return 0;
    } else {
        printf("\n>>> SOME TESTS FAILED <<<\n");
        return 1;
    }
}
