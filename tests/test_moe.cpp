// ============================================================================
// MoE 三算子端到端 NPU 测试
// ----------------------------------------------------------------------------
// 测试流程:
//   1. ACL 初始化与设备管理
//   2. 构造 MoE 输入数据 (FP16)
//   3. CPU 参考实现计算期望输出 (FP32)
//   4. 依次调用 3 个 MoE 算子 (aclrtLaunchKernel):
//      a. MoeRoutingTopK:  x + w_gate → expert_ids + weights
//      b. MoeExpertFFN:    x + expert_ids + weights + expert_weights + workspace → expert_out
//      c. MoeCombine:      expert_out + expert_ids + weights → y
//   5. 精度校验 (max_abs_error < 0.05, cosine_similarity > 0.99)
//   6. 性能计时 (多次运行取平均)
//   7. 资源释放
//
// 运行前提: 已通过 build.sh 编译并部署算子包 (custom_opp_*.run)
// ============================================================================

#include <acl/acl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

// ============================================================================
// 外部 Kernel 函数声明 (由 op_kernel 编译产生)
// ============================================================================
extern "C" {
// MoeRoutingTopK: x + wGate → expertIds + weights
void MoeRoutingTopK(void* x, void* wGate, void* expertIds, void* weights, void* tiling);
// MoeExpertFFN: x + expertIds + weights + expertWeights + workspace → expertOut
void MoeExpertFFN(void* x, void* expertIds, void* weights, void* expertWeights,
                  void* expertOut, void* workspace, void* tiling);
// MoeCombine: expertOut + expertIds + weights → y
void MoeCombine(void* expertOut, void* expertIds, void* weights, void* y, void* tiling);
}

// ============================================================================
// 测试参数配置
// ============================================================================
struct TestConfig {
    int32_t numTokens;
    int32_t hiddenSize;
    int32_t numExperts;
    int32_t topK;
    int32_t intermediateSize;
    const char* name;
};

// ============================================================================
// Tiling 数据结构 (Host 侧定义，与 kernel 侧 GET_TILING_DATA 反序列化匹配)
// ============================================================================
// 注意: 这些结构体以纯字段顺序排列，与 CANN TILING_DATA_FIELD_DEF 宏
// 生成的序列化布局一致。如 CANN 版本不同导致布局变化，需相应调整。

// MoeRoutingTopK Tiling 参数
struct RoutingTiling {
    uint32_t numTokens;
    uint32_t hiddenSize;
    uint32_t numExperts;
    uint32_t topK;
    uint32_t tileS;
    uint32_t blockDim;
    uint32_t totalTaskNum;
    uint32_t coreTaskNum;
};

// MoeExpertFFN Tiling 参数
struct ExpertFFNTiling {
    uint32_t numTokens;
    uint32_t hiddenSize;
    uint32_t intermediateSize;
    uint32_t numExperts;
    uint32_t topK;
    uint32_t tileTokens;
    uint32_t tileIntermediate;
    uint32_t tileK;
    uint32_t blockDim;
    uint32_t totalTaskNum;
};

// MoeCombine Tiling 参数
struct CombineTiling {
    uint32_t numTokens;
    uint32_t hiddenSize;
    uint32_t topK;
    uint32_t tileS;
    uint32_t blockDim;
    uint32_t totalTaskNum;
};

// ============================================================================
// 编译期常量 (与 moe_ops.cpp 中一致)
// ============================================================================
static constexpr uint32_t DEFAULT_BLOCK_DIM = 24;    // 910B 共 24 个 AI Core
static constexpr uint32_t MAX_TILE_BATCHES  = 32;    // ExpertFFN workspace 常数
static constexpr uint32_t ROUTING_TILE_S    = 32;    // Routing 分块大小
static constexpr uint32_t FFN_TILE_TOKENS   = 16;    // FFN token 分块
static constexpr uint32_t FFN_TILE_INTER    = 256;   // FFN intermediate 分块
static constexpr uint32_t FFN_TILE_K        = 128;   // FFN K 维分块
static constexpr uint32_t COMBINE_TILE_S    = 64;    // Combine 分块大小

// ============================================================================
// FP16 转换辅助函数
// ============================================================================
static inline uint16_t FloatToHalf(float f)
{
    return aclFloatToFloat16(f);
}

static inline float HalfToFloat(uint16_t h)
{
    return aclFloat16ToFloat(h);
}

// ============================================================================
// Tiling 参数计算 (与 host 侧 TilingFunc 逻辑一致)
// ============================================================================

// 计算 MoeRoutingTopK 的 Tiling 参数
RoutingTiling ComputeRoutingTiling(int32_t numTokens, int32_t hiddenSize,
                                    int32_t numExperts, int32_t topK)
{
    RoutingTiling t{};
    t.numTokens  = static_cast<uint32_t>(numTokens);
    t.hiddenSize = static_cast<uint32_t>(hiddenSize);
    t.numExperts = static_cast<uint32_t>(numExperts);
    t.topK       = static_cast<uint32_t>(topK);
    t.blockDim   = DEFAULT_BLOCK_DIM;

    // UB 约束: W_gate(常驻) + x_tile + logits_tile <= 256KB
    // W_gate = hiddenSize * numExperts * 2 bytes (FP16)
    const uint32_t wGateBytes = t.hiddenSize * t.numExperts * 2;
    const uint32_t remainUB   = 256 * 1024 - wGateBytes;
    const uint32_t perToken   = t.hiddenSize * 2 + t.numExperts * 4;
    uint32_t tileS = remainUB / perToken;
    tileS = (tileS / 16) * 16;
    if (tileS == 0) tileS = 1;
    if (tileS > ROUTING_TILE_S) tileS = ROUTING_TILE_S;
    if (tileS > t.numTokens) tileS = t.numTokens;

    t.tileS        = tileS;
    t.totalTaskNum = (t.numTokens + tileS - 1) / tileS;
    t.coreTaskNum  = (t.totalTaskNum + t.blockDim - 1) / t.blockDim;
    return t;
}

// 计算 MoeExpertFFN 的 Tiling 参数与 workspace 大小
ExpertFFNTiling ComputeExpertFFNTiling(int32_t numTokens, int32_t hiddenSize,
                                        int32_t intermediateSize,
                                        int32_t numExperts, int32_t topK,
                                        size_t& workspaceBytes)
{
    ExpertFFNTiling t{};
    t.numTokens        = static_cast<uint32_t>(numTokens);
    t.hiddenSize       = static_cast<uint32_t>(hiddenSize);
    t.intermediateSize = static_cast<uint32_t>(intermediateSize);
    t.numExperts       = static_cast<uint32_t>(numExperts);
    t.topK             = static_cast<uint32_t>(topK);
    t.tileTokens       = FFN_TILE_TOKENS;
    t.tileIntermediate = FFN_TILE_INTER;
    t.tileK            = FFN_TILE_K;
    t.blockDim         = DEFAULT_BLOCK_DIM;
    t.totalTaskNum     = t.numExperts;  // 每专家一个任务

    // workspace = numExperts * MAX_TILE_BATCHES * tileTokens * intermediateSize * sizeof(half)
    workspaceBytes = static_cast<size_t>(t.numExperts) * MAX_TILE_BATCHES *
                     t.tileTokens * t.intermediateSize * sizeof(uint16_t);
    return t;
}

// 计算 MoeCombine 的 Tiling 参数
CombineTiling ComputeCombineTiling(int32_t numTokens, int32_t hiddenSize, int32_t topK)
{
    CombineTiling t{};
    t.numTokens  = static_cast<uint32_t>(numTokens);
    t.hiddenSize = static_cast<uint32_t>(hiddenSize);
    t.topK       = static_cast<uint32_t>(topK);
    t.blockDim   = DEFAULT_BLOCK_DIM;

    uint32_t tileS = COMBINE_TILE_S;
    if (tileS > t.numTokens) tileS = t.numTokens;

    t.tileS        = tileS;
    t.totalTaskNum = (t.numTokens + tileS - 1) / tileS;
    return t;
}

// ============================================================================
// CPU 参考实现: NaiveMoE (FP32 精度)
// ----------------------------------------------------------------------------
// 输入:
//   x:              [numTokens, hiddenSize]
//   w_gate:         [hiddenSize, numExperts]
//   expert_weights: [numExperts, 3*intermediateSize, hiddenSize]
//     每个专家: W_gate[I,H] | W_up[I,H] | W_down[H,I]  (行优先)
// 输出:
//   output:         [numTokens, hiddenSize]
// ============================================================================
static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

void NaiveMoE(const std::vector<float>& x,
              const std::vector<float>& w_gate,
              const std::vector<float>& expert_weights,
              int T, int H, int E, int K, int I,
              std::vector<float>& output)
{
    output.assign(static_cast<size_t>(T) * H, 0.0f);

    // Step 1: Gating — logits = x @ w_gate → [T, E]
    std::vector<float> logits(static_cast<size_t>(T) * E, 0.0f);
    for (int t = 0; t < T; t++) {
        for (int e = 0; e < E; e++) {
            float dot = 0.0f;
            for (int h = 0; h < H; h++)
                dot += x[t * H + h] * w_gate[h * E + e];
            logits[t * E + e] = dot;
        }
    }

    // Step 2: Softmax — 逐行做数值稳定的 softmax
    std::vector<float> probs(static_cast<size_t>(T) * E);
    for (int t = 0; t < T; t++) {
        float maxVal = -1e30f;
        for (int e = 0; e < E; e++)
            if (logits[t * E + e] > maxVal) maxVal = logits[t * E + e];
        float sumExp = 0.0f;
        for (int e = 0; e < E; e++) {
            probs[t * E + e] = std::exp(logits[t * E + e] - maxVal);
            sumExp += probs[t * E + e];
        }
        for (int e = 0; e < E; e++)
            probs[t * E + e] /= sumExp;
    }

    // Step 3: TopK — 对每行选 Top-K 专家 (与 kernel 一致，做权重归一化)
    std::vector<int>   expert_ids(static_cast<size_t>(T) * K);
    std::vector<float> weights(static_cast<size_t>(T) * K);
    for (int t = 0; t < T; t++) {
        std::vector<int> indices(E);
        std::iota(indices.begin(), indices.end(), 0);
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

    // Step 4: Expert FFN — SwiGLU (gate + up → SiLU → mul → down)
    std::vector<float> ffn_out(static_cast<size_t>(T) * K * H, 0.0f);
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            int e = expert_ids[t * K + k];
            size_t expertBase = static_cast<size_t>(e) * 3 * I * H;
            const float* W_gate_e = &expert_weights[expertBase];
            const float* W_up_e   = &expert_weights[expertBase + static_cast<size_t>(I) * H];
            const float* W_down_e = &expert_weights[expertBase + static_cast<size_t>(2) * I * H];

            // gate_proj[i] = x @ W_gate_e[i, :]^T
            std::vector<float> gate_proj(I), up_proj(I), hidden(I);
            for (int i = 0; i < I; i++) {
                float gDot = 0.0f, uDot = 0.0f;
                for (int h = 0; h < H; h++) {
                    float xVal = x[t * H + h];
                    gDot += xVal * W_gate_e[i * H + h];
                    uDot += xVal * W_up_e[i * H + h];
                }
                gate_proj[i] = gDot;
                up_proj[i]   = uDot;
            }

            // SwiGLU: hidden = SiLU(gate) * up
            for (int i = 0; i < I; i++)
                hidden[i] = silu(gate_proj[i]) * up_proj[i];

            // down_proj: ffn_out[h] = hidden @ W_down_e[h, :]^T
            for (int h = 0; h < H; h++) {
                float dot = 0.0f;
                for (int i = 0; i < I; i++)
                    dot += hidden[i] * W_down_e[h * I + i];
                ffn_out[t * K * H + k * H + h] = dot;
            }
        }
    }

    // Step 5: Combine — output[t] = Σ_k weight_k * ffn_out_k
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            float w = weights[t * K + k];
            for (int h = 0; h < H; h++)
                output[t * H + h] += w * ffn_out[t * K * H + k * H + h];
        }
    }
}

// ============================================================================
// 精度比较: 最大绝对误差、平均绝对误差、余弦相似度
// ============================================================================
void CompareResults(const std::vector<float>& actual,
                    const std::vector<float>& expected,
                    float& maxAbsErr, float& meanAbsErr, double& cosSim)
{
    maxAbsErr = 0.0f;
    float sumAbsErr = 0.0f;
    double dotAB = 0.0, dotAA = 0.0, dotBB = 0.0;

    for (size_t i = 0; i < actual.size(); i++) {
        float err = std::abs(actual[i] - expected[i]);
        sumAbsErr += err;
        if (err > maxAbsErr) maxAbsErr = err;
        dotAB += static_cast<double>(actual[i]) * static_cast<double>(expected[i]);
        dotAA += static_cast<double>(actual[i]) * static_cast<double>(actual[i]);
        dotBB += static_cast<double>(expected[i]) * static_cast<double>(expected[i]);
    }
    meanAbsErr = sumAbsErr / static_cast<float>(actual.size());
    cosSim = (dotAA > 0 && dotBB > 0) ? dotAB / (std::sqrt(dotAA) * std::sqrt(dotBB)) : 0.0;
}

// ============================================================================
// ACL 辅助宏: 错误检查
// ============================================================================
#define ACL_CHECK(call)                                                       \
    do {                                                                      \
        aclError _ret = (call);                                               \
        if (_ret != ACL_SUCCESS) {                                            \
            printf("[ERROR] %s 失败, ret=%d, 行号=%d\n", #call, _ret, __LINE__); \
            goto CLEANUP;                                                     \
        }                                                                     \
    } while (0)

// ============================================================================
// 执行单次 MoE 端到端测试
// ============================================================================
bool RunMoETest(const TestConfig& cfg)
{
    const int32_t T = cfg.numTokens;
    const int32_t H = cfg.hiddenSize;
    const int32_t E = cfg.numExperts;
    const int32_t K = cfg.topK;
    const int32_t I = cfg.intermediateSize;

    printf("\n========================================\n");
    printf(" 测试: %s\n", cfg.name);
    printf("   T=%d H=%d E=%d K=%d I=%d\n", T, H, E, K, I);
    printf("========================================\n");

    // ---- 1. 设备指针声明 (初始化为 nullptr) ----
    void *xDev = nullptr, *wGateDev = nullptr, *expertWeightsDev = nullptr;
    void *expertIdsDev = nullptr, *weightsDev = nullptr;
    void *expertOutDev = nullptr, *workspaceDev = nullptr, *yDev = nullptr;
    void *routingTilingDev = nullptr, *ffnTilingDev = nullptr, *combineTilingDev = nullptr;
    aclrtStream stream = nullptr;

    // ---- 2. 生成随机 FP32 输入数据 (CPU 侧) ----
    const size_t xSize    = static_cast<size_t>(T) * H;
    const size_t wgSize   = static_cast<size_t>(H) * E;
    const size_t ewSize   = static_cast<size_t>(E) * 3 * I * H;
    const size_t eidSize  = static_cast<size_t>(T) * K;
    const size_t wSize    = static_cast<size_t>(T) * K;
    const size_t eoSize   = static_cast<size_t>(T) * K * H;
    const size_t ySize    = static_cast<size_t>(T) * H;

    srand(42);
    auto randFloat = []() -> float {
        return (rand() / static_cast<float>(RAND_MAX)) * 2.0f - 1.0f;
    };

    std::vector<float> xHost(xSize), wGateHost(wgSize), expertWeightsHost(ewSize);
    for (size_t i = 0; i < xSize; i++) xHost[i] = randFloat();
    for (size_t i = 0; i < wgSize; i++) wGateHost[i] = randFloat();
    for (size_t i = 0; i < ewSize; i++) expertWeightsHost[i] = randFloat();

    // ---- 3. CPU 参考实现计算期望输出 ----
    std::vector<float> expectedOut;
    NaiveMoE(xHost, wGateHost, expertWeightsHost, T, H, E, K, I, expectedOut);
    printf("[INFO] CPU 参考实现计算完成\n");

    // ---- 4. 转换输入为 FP16 ----
    std::vector<uint16_t> xHalf(xSize), wGateHalf(wgSize), expertWeightsHalf(ewSize);
    for (size_t i = 0; i < xSize; i++) xHalf[i] = FloatToHalf(xHost[i]);
    for (size_t i = 0; i < wgSize; i++) wGateHalf[i] = FloatToHalf(wGateHost[i]);
    for (size_t i = 0; i < ewSize; i++) expertWeightsHalf[i] = FloatToHalf(expertWeightsHost[i]);

    // ---- 5. 计算 Tiling 参数 ----
    auto routingTiling = ComputeRoutingTiling(T, H, E, K);
    size_t ffnWsBytes = 0;
    auto ffnTiling = ComputeExpertFFNTiling(T, H, I, E, K, ffnWsBytes);
    auto combineTiling = ComputeCombineTiling(T, H, K);

    printf("[INFO] Routing Tiling:  tileS=%u tasks=%u\n", routingTiling.tileS, routingTiling.totalTaskNum);
    printf("[INFO] ExpertFFN Tiling: tileTokens=%u tileInter=%u tileK=%u tasks=%u workspace=%zuB\n",
           ffnTiling.tileTokens, ffnTiling.tileIntermediate, ffnTiling.tileK,
           ffnTiling.totalTaskNum, ffnWsBytes);
    printf("[INFO] Combine Tiling:   tileS=%u tasks=%u\n", combineTiling.tileS, combineTiling.totalTaskNum);

    // ---- 6. 分配 NPU 设备内存 ----
    const size_t xBytes    = xSize * sizeof(uint16_t);
    const size_t wgBytes   = wgSize * sizeof(uint16_t);
    const size_t ewBytes   = ewSize * sizeof(uint16_t);
    const size_t eidBytes  = eidSize * sizeof(int32_t);
    const size_t wBytes    = wSize * sizeof(uint16_t);
    const size_t eoBytes   = eoSize * sizeof(uint16_t);
    const size_t yBytes    = ySize * sizeof(uint16_t);

    ACL_CHECK(aclrtMalloc(&xDev, xBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&wGateDev, wgBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&expertWeightsDev, ewBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&expertIdsDev, eidBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&weightsDev, wBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&expertOutDev, eoBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&workspaceDev, ffnWsBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&yDev, yBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    // 分配 Tiling 数据设备内存
    ACL_CHECK(aclrtMalloc(&routingTilingDev, sizeof(RoutingTiling), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&ffnTilingDev, sizeof(ExpertFFNTiling), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&combineTilingDev, sizeof(CombineTiling), ACL_MEM_MALLOC_HUGE_FIRST));
    printf("[INFO] 设备内存分配完成\n");

    // ---- 7. 拷贝输入数据和 Tiling 到设备 ----
    ACL_CHECK(aclrtMemcpy(xDev, xBytes, xHalf.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(wGateDev, wgBytes, wGateHalf.data(), wgBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(expertWeightsDev, ewBytes, expertWeightsHalf.data(), ewBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(routingTilingDev, sizeof(RoutingTiling), &routingTiling, sizeof(RoutingTiling), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(ffnTilingDev, sizeof(ExpertFFNTiling), &ffnTiling, sizeof(ExpertFFNTiling), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(combineTilingDev, sizeof(CombineTiling), &combineTiling, sizeof(CombineTiling), ACL_MEMCPY_HOST_TO_DEVICE));
    printf("[INFO] 输入数据已搬移至 NPU\n");

    // ---- 8. 创建 Stream ----
    ACL_CHECK(aclrtCreateStream(&stream));

    // ---- 9. 依次调用 3 个 MoE 算子 ----

    // 9.1 MoeRoutingTopK: x + w_gate → expert_ids + weights
    printf("[INFO] 调用 MoeRoutingTopK ...\n");
    ACL_CHECK(aclrtLaunchKernel(MoeRoutingTopK, stream, routingTiling.blockDim,
        xDev, wGateDev, expertIdsDev, weightsDev, routingTilingDev));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    printf("[INFO] MoeRoutingTopK 执行完成\n");

    // 9.2 MoeExpertFFN: x + expert_ids + weights + expert_weights + workspace → expert_out
    printf("[INFO] 调用 MoeExpertFFN ...\n");
    ACL_CHECK(aclrtLaunchKernel(MoeExpertFFN, stream, ffnTiling.blockDim,
        xDev, expertIdsDev, weightsDev, expertWeightsDev, expertOutDev,
        workspaceDev, ffnTilingDev));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    printf("[INFO] MoeExpertFFN 执行完成\n");

    // 9.3 MoeCombine: expert_out + expert_ids + weights → y
    printf("[INFO] 调用 MoeCombine ...\n");
    ACL_CHECK(aclrtLaunchKernel(MoeCombine, stream, combineTiling.blockDim,
        expertOutDev, expertIdsDev, weightsDev, yDev, combineTilingDev));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    printf("[INFO] MoeCombine 执行完成\n");

    // ---- 10. 拷回 NPU 输出结果 ----
    std::vector<uint16_t> yHalf(ySize);
    ACL_CHECK(aclrtMemcpy(yHalf.data(), yBytes, yDev, yBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> actualOut(ySize);
    for (size_t i = 0; i < ySize; i++)
        actualOut[i] = HalfToFloat(yHalf[i]);

    // ---- 11. 精度校验 ----
    float maxAbsErr = 0.0f, meanAbsErr = 0.0f;
    double cosSim = 0.0;
    CompareResults(actualOut, expectedOut, maxAbsErr, meanAbsErr, cosSim);

    printf("\n========================================\n");
    printf(" 精度验证结果 (%s)\n", cfg.name);
    printf("   最大绝对误差: %.6f  (阈值 < 0.05)\n", maxAbsErr);
    printf("   平均绝对误差: %.6f  (阈值 < 0.01)\n", meanAbsErr);
    printf("   余弦相似度:   %.10f  (阈值 > 0.99)\n", cosSim);
    printf("========================================\n");

    const float MAX_ABS_THRESH = 0.05f;
    const float MEAN_ABS_THRESH = 0.01f;
    const double COS_SIM_THRESH = 0.99;
    bool passed = (maxAbsErr < MAX_ABS_THRESH && meanAbsErr < MEAN_ABS_THRESH
                   && cosSim > COS_SIM_THRESH);

    if (passed) {
        printf("[PASS] 精度验证通过!\n");
    } else {
        printf("[FAIL] 精度超差! (max<%.4f, mean<%.4f, cosSim>%.2f)\n",
               MAX_ABS_THRESH, MEAN_ABS_THRESH, COS_SIM_THRESH);
    }

CLEANUP:
    // ---- 12. 释放资源 ----
    if (stream)              aclrtDestroyStream(stream);
    if (combineTilingDev)    aclrtFree(combineTilingDev);
    if (ffnTilingDev)        aclrtFree(ffnTilingDev);
    if (routingTilingDev)    aclrtFree(routingTilingDev);
    if (yDev)                aclrtFree(yDev);
    if (workspaceDev)        aclrtFree(workspaceDev);
    if (expertOutDev)        aclrtFree(expertOutDev);
    if (weightsDev)          aclrtFree(weightsDev);
    if (expertIdsDev)        aclrtFree(expertIdsDev);
    if (expertWeightsDev)    aclrtFree(expertWeightsDev);
    if (wGateDev)            aclrtFree(wGateDev);
    if (xDev)                aclrtFree(xDev);

    return passed;
}

// ============================================================================
// 性能计时: 对目标规模多次运行取平均
// ============================================================================
bool RunPerfTest(const TestConfig& cfg, int numRuns = 10)
{
    const int32_t T = cfg.numTokens;
    const int32_t H = cfg.hiddenSize;
    const int32_t E = cfg.numExperts;
    const int32_t K = cfg.topK;
    const int32_t I = cfg.intermediateSize;

    printf("\n========================================\n");
    printf(" 性能测试: %s (%d 次运行)\n", cfg.name, numRuns);
    printf("   T=%d H=%d E=%d K=%d I=%d\n", T, H, E, K, I);
    printf("========================================\n");

    void *xDev = nullptr, *wGateDev = nullptr, *expertWeightsDev = nullptr;
    void *expertIdsDev = nullptr, *weightsDev = nullptr;
    void *expertOutDev = nullptr, *workspaceDev = nullptr, *yDev = nullptr;
    void *routingTilingDev = nullptr, *ffnTilingDev = nullptr, *combineTilingDev = nullptr;
    aclrtStream stream = nullptr;
    bool perfOk = true;

    // 分配内存 (用随机数据初始化，不关心精度)
    const size_t xSize   = static_cast<size_t>(T) * H;
    const size_t wgSize  = static_cast<size_t>(H) * E;
    const size_t ewSize  = static_cast<size_t>(E) * 3 * I * H;
    const size_t eidSize = static_cast<size_t>(T) * K;
    const size_t wSize   = static_cast<size_t>(T) * K;
    const size_t eoSize  = static_cast<size_t>(T) * K * H;
    const size_t ySize   = static_cast<size_t>(T) * H;

    std::vector<uint16_t> xHalf(xSize, FloatToHalf(0.01f));
    std::vector<uint16_t> wGateHalf(wgSize, FloatToHalf(0.01f));
    std::vector<uint16_t> ewHalf(ewSize, FloatToHalf(0.01f));

    auto routingTiling = ComputeRoutingTiling(T, H, E, K);
    size_t ffnWsBytes = 0;
    auto ffnTiling = ComputeExpertFFNTiling(T, H, I, E, K, ffnWsBytes);
    auto combineTiling = ComputeCombineTiling(T, H, K);

    aclrtMalloc(&xDev, xSize * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&wGateDev, wgSize * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&expertWeightsDev, ewSize * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&expertIdsDev, eidSize * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&weightsDev, wSize * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&expertOutDev, eoSize * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&workspaceDev, ffnWsBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&yDev, ySize * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&routingTilingDev, sizeof(RoutingTiling), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&ffnTilingDev, sizeof(ExpertFFNTiling), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&combineTilingDev, sizeof(CombineTiling), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(xDev, xSize * 2, xHalf.data(), xSize * 2, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(wGateDev, wgSize * 2, wGateHalf.data(), wgSize * 2, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(expertWeightsDev, ewSize * 2, ewHalf.data(), ewSize * 2, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(routingTilingDev, sizeof(RoutingTiling), &routingTiling, sizeof(RoutingTiling), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(ffnTilingDev, sizeof(ExpertFFNTiling), &ffnTiling, sizeof(ExpertFFNTiling), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(combineTilingDev, sizeof(CombineTiling), &combineTiling, sizeof(CombineTiling), ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtCreateStream(&stream);

    // 预热一次
    aclrtLaunchKernel(MoeRoutingTopK, stream, routingTiling.blockDim,
        xDev, wGateDev, expertIdsDev, weightsDev, routingTilingDev);
    aclrtLaunchKernel(MoeExpertFFN, stream, ffnTiling.blockDim,
        xDev, expertIdsDev, weightsDev, expertWeightsDev, expertOutDev,
        workspaceDev, ffnTilingDev);
    aclrtLaunchKernel(MoeCombine, stream, combineTiling.blockDim,
        expertOutDev, expertIdsDev, weightsDev, yDev, combineTilingDev);
    aclrtSynchronizeStream(stream);

    // 多次运行计时
    double totalMs = 0.0;
    for (int run = 0; run < numRuns; run++) {
        auto t0 = std::chrono::high_resolution_clock::now();

        aclrtLaunchKernel(MoeRoutingTopK, stream, routingTiling.blockDim,
            xDev, wGateDev, expertIdsDev, weightsDev, routingTilingDev);
        aclrtLaunchKernel(MoeExpertFFN, stream, ffnTiling.blockDim,
            xDev, expertIdsDev, weightsDev, expertWeightsDev, expertOutDev,
            workspaceDev, ffnTilingDev);
        aclrtLaunchKernel(MoeCombine, stream, combineTiling.blockDim,
            expertOutDev, expertIdsDev, weightsDev, yDev, combineTilingDev);
        aclrtSynchronizeStream(stream);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        totalMs += ms;
        printf("  Run %2d: %.3f ms\n", run + 1, ms);
    }

    double avgMs = totalMs / numRuns;
    printf("\n[PERF] 端到端平均耗时: %.3f ms (%d 次平均)\n", avgMs, numRuns);
    printf("[PERF] 吞吐量: %.2f tokens/s\n",
           static_cast<double>(T) / (avgMs / 1000.0));

    // 释放资源
    if (stream)              aclrtDestroyStream(stream);
    if (combineTilingDev)    aclrtFree(combineTilingDev);
    if (ffnTilingDev)        aclrtFree(ffnTilingDev);
    if (routingTilingDev)    aclrtFree(routingTilingDev);
    if (yDev)                aclrtFree(yDev);
    if (workspaceDev)        aclrtFree(workspaceDev);
    if (expertOutDev)        aclrtFree(expertOutDev);
    if (weightsDev)          aclrtFree(weightsDev);
    if (expertIdsDev)        aclrtFree(expertIdsDev);
    if (expertWeightsDev)    aclrtFree(expertWeightsDev);
    if (wGateDev)            aclrtFree(wGateDev);
    if (xDev)                aclrtFree(xDev);

    return perfOk;
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char* argv[])
{
    printf("========================================\n");
    printf(" MoE 三算子端到端 NPU 测试\n");
    printf("   MoeRoutingTopK → MoeExpertFFN → MoeCombine\n");
    printf("========================================\n");

    // ---- ACL 初始化 ----
    const int32_t deviceId = 0;
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
        printf("[ERROR] aclInit 失败, ret=%d\n", ret);
        return -1;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        printf("[ERROR] aclrtSetDevice 失败, ret=%d\n", ret);
        aclFinalize();
        return -1;
    }
    printf("[INFO] ACL 初始化成功, deviceId=%d\n", deviceId);

    // ---- 测试用例定义 ----
    // 小规模: T=8, H=64, E=4, K=2, I=256
    TestConfig smallCfg = {8, 64, 4, 2, 256, "小规模 T=8 H=64 E=4 K=2 I=256"};
    // 目标规模: T=128, H=2048, E=16, K=2, I=8192
    TestConfig targetCfg = {128, 2048, 16, 2, 8192, "目标规模 T=128 H=2048 E=16 K=2 I=8192"};

    int passCount = 0;
    int totalCount = 2;

    // ---- 小规模精度测试 ----
    if (RunMoETest(smallCfg)) passCount++;

    // ---- 目标规模精度测试 ----
    if (RunMoETest(targetCfg)) passCount++;

    // ---- 目标规模性能测试 ----
    printf("\n[INFO] 开始性能计时测试 ...\n");
    RunPerfTest(targetCfg, 10);

    // ---- 汇总 ----
    printf("\n========================================\n");
    printf(" 测试汇总\n");
    printf("========================================\n");
    printf(" 精度测试: %d/%d PASS\n", passCount, totalCount);
    if (passCount == totalCount) {
        printf("\n>>> ALL TESTS PASSED <<<\n");
    } else {
        printf("\n>>> SOME TESTS FAILED <<<\n");
    }

    // ---- ACL 清理 ----
    aclrtResetDevice(deviceId);
    aclFinalize();
    printf("[INFO] 资源已释放, 测试结束\n");

    return (passCount == totalCount) ? 0 : 1;
}
