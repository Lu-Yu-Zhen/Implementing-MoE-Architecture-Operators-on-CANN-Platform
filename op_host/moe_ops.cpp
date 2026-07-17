// ============================================================================
// MoE 算子 Host 侧实现
// ----------------------------------------------------------------------------
// 包含 3 个子算子的原型注册(REG_OP)、InferShape、InferDataType、TilingFunc:
//   1. MoeRoutingTopK  - 门控路由 + Top-K 专家选择
//   2. MoeExpertFFN    - 专家 FFN 前向 (GEMM + SwiGLU + GEMM)
//   3. MoeCombine      - 加权合并专家输出
//
// MoE 流水线:
//   x → [MoeRoutingTopK] → expert_ids, weights
//     → [MoeExpertFFN]   → expert_out[numTokens, topK, hiddenSize]
//     → [MoeCombine]     → y[numTokens, hiddenSize]
//
// 目标硬件: Ascend 910B (24 AI Cores, ~256KB UB/core)
// 数据类型: FP16 (GEMM 输出 FP32 累加)
// ============================================================================

#include "moe_tiling.h"

#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "error_log.h"
#include "tiling/platform_info_ascendc.h"

#include <cmath>
#include <algorithm>

namespace optiling {

// ============================================================================
// 编译期常量：MoE 默认参数
// ============================================================================
namespace {
constexpr uint32_t DEFAULT_NUM_EXPERTS    = 16;
constexpr uint32_t DEFAULT_TOP_K          = 2;
constexpr uint32_t DEFAULT_HIDDEN_SIZE    = 2048;
constexpr uint32_t DEFAULT_INTERMEDIATE   = 8192;
constexpr uint32_t DEFAULT_BLOCK_DIM      = 24;   // 910B 共 24 个 AI Core
constexpr uint32_t UB_SIZE_BYTES          = 256 * 1024;  // 256KB UB/core

// MoeRoutingTopK 分块参数
// W_gate 常驻: hiddenSize * numExperts * 2 = 2048*16*2 = 64KB
// 剩余 UB: 256KB - 64KB = 192KB
// tileS = min(numTokens, 192*1024 / (2048*2 + 16*4)) ≈ 45 → 取 32
constexpr uint32_t ROUTING_TILE_S         = 32;

// MoeExpertFFN 分块参数
// 两阶段设计 (tileTokens=16):
// Stage A (SwiGLU): xBuf(64KB) + wTileBuf(64KB) + projBuf(16KB) + interBuf(8KB) + tempBuf(16KB) ≈ 170KB
// Stage B (Down):   interBuf(8KB) + wTileBuf(64KB) + projBuf(16KB) + tempBuf(16KB) ≈ 104KB
// 两阶段不同时执行，取 max ≈ 170KB < 256KB
constexpr uint32_t FFN_TILE_TOKENS        = 16;
constexpr uint32_t FFN_TILE_INTERMEDIATE  = 256;
constexpr uint32_t FFN_TILE_K             = 128;

// MoeCombine 分块参数
constexpr uint32_t COMBINE_TILE_S         = 64;
}  // namespace

// ============================================================================
// MoeRoutingTopK TilingFunc
// ============================================================================
static ge::graphStatus MoeRoutingTilingFunc(gert::TilingContext* context)
{
    OP_LOGI(context->GetNodeName(), "MoeRoutingTopK TilingFunc begin.");

    // ---- 1. 获取输入 shape ----
    auto xShape      = context->GetInputShape(0);  // x: [numTokens, hiddenSize]
    auto wGateShape  = context->GetInputShape(1);  // w_gate: [hiddenSize, numExperts]

    if (xShape == nullptr || wGateShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "输入 shape 为空，请检查算子输入是否正确连接。");
        return ge::GRAPH_FAILED;
    }

    const auto& xDims = xShape->GetOriginShape().GetDims();
    const auto& wDims = wGateShape->GetOriginShape().GetDims();

    // ---- 2. shape 合法性校验 ----
    if (xDims.size() != 2 || wDims.size() != 2) {
        OP_LOGE(context->GetNodeName(),
                "输入维度不正确: x dims=%zu (期望2), w_gate dims=%zu (期望2)。",
                xDims.size(), wDims.size());
        return ge::GRAPH_FAILED;
    }
    if (xDims[1] != wDims[0]) {
        OP_LOGE(context->GetNodeName(),
                "hiddenSize 不匹配: x[%ld] vs w_gate[%ld]。", xDims[1], wDims[0]);
        return ge::GRAPH_FAILED;
    }

    // ---- 3. 提取参数 ----
    const uint32_t numTokens  = static_cast<uint32_t>(xDims[0]);
    const uint32_t hiddenSize = static_cast<uint32_t>(xDims[1]);
    const uint32_t numExperts = static_cast<uint32_t>(wDims[1]);

    // 从属性读取 topK
    uint32_t topK = DEFAULT_TOP_K;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrsNumber() > 0) {
        const int64_t* topKAttr = attrs->GetAttrPointer<int64_t>(0);
        if (topKAttr != nullptr && *topKAttr > 0) {
            topK = static_cast<uint32_t>(*topKAttr);
        }
    }

    // ---- 4. 获取平台信息 ----
    uint32_t blockDim = DEFAULT_BLOCK_DIM;
    auto platformInfo = platform_ascendc::PlatformInfoAscendC(context->GetPlatformInfo());
    if (platformInfo.IsSet()) {
        auto aiCoreSpec = platformInfo.GetCoreSpecAic();
        uint32_t coreNum = aiCoreSpec.vectorCoreNum;
        if (coreNum > 0) {
            blockDim = coreNum;
        }
        OP_LOGI(context->GetNodeName(),
                "平台信息: vectorCoreNum=%u, ubSize=%lu Bytes。",
                coreNum, aiCoreSpec.ubSize);
    } else {
        OP_LOGW(context->GetNodeName(), "未获取到平台信息，使用默认参数 blockDim=%u。", blockDim);
    }

    // ---- 5. 计算分块参数 ----
    // UB 约束: W_gate(常驻) + x_tile + logits_tile <= UB_SIZE
    // W_gate = hiddenSize * numExperts * 2 bytes
    // x_tile = tileS * hiddenSize * 2 bytes
    // logits_tile = tileS * numExperts * 4 bytes (FP32)
    const uint32_t wGateBytes = hiddenSize * numExperts * 2;
    const uint32_t remainUB = UB_SIZE_BYTES - wGateBytes;
    const uint32_t bytesPerToken = hiddenSize * 2 + numExperts * 4;
    uint32_t tileS = remainUB / bytesPerToken;
    // 对齐到 16 的倍数
    tileS = (tileS / 16) * 16;
    if (tileS == 0) tileS = 1;
    if (tileS > ROUTING_TILE_S) tileS = ROUTING_TILE_S;
    if (tileS > numTokens) tileS = numTokens;

    const uint32_t totalTaskNum = (numTokens + tileS - 1) / tileS;
    const uint32_t coreTaskNum = (totalTaskNum + blockDim - 1) / blockDim;

    // ---- 6. 填充 TilingData ----
    MoeRoutingTilingData tiling;
    tiling.set_numTokens(numTokens);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_numExperts(numExperts);
    tiling.set_topK(topK);
    tiling.set_tileS(tileS);
    tiling.set_blockDim(blockDim);
    tiling.set_totalTaskNum(totalTaskNum);
    tiling.set_coreTaskNum(coreTaskNum);

    // ---- 7. 设置 BlockDim ----
    context->SetBlockDim(blockDim);

    // ---- 8. 序列化 TilingData ----
    if (!tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                              context->GetRawTilingData()->GetCapacity())) {
        OP_LOGE(context->GetNodeName(), "TilingData 序列化失败，缓冲区容量不足。");
        return ge::GRAPH_FAILED;
    }
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // ---- 9. Workspace 设置 ----
    auto workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes != nullptr) {
        workspaceSizes[0] = 0;
    }

    OP_LOGI(context->GetNodeName(),
            "Tiling 完成: numTokens=%u hiddenSize=%u numExperts=%u topK=%u "
            "tileS=%u totalTaskNum=%u blockDim=%u coreTaskNum=%u",
            numTokens, hiddenSize, numExperts, topK,
            tileS, totalTaskNum, blockDim, coreTaskNum);

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MoeRoutingTopK InferShape
// ============================================================================
static ge::graphStatus MoeRoutingInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& xDims = xShape->GetOriginShape().GetDims();
    if (xDims.size() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int64_t numTokens = xDims[0];
    int64_t topK = DEFAULT_TOP_K;

    // 从属性读取 topK（如果可用）
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrsNumber() > 0) {
        const int64_t* topKAttr = attrs->GetAttrPointer<int64_t>(0);
        if (topKAttr != nullptr && *topKAttr > 0) {
            topK = *topKAttr;
        }
    }

    // output 0: expert_ids [numTokens, topK] INT32
    gert::Shape* expertIdsShape = context->GetOutputShape(0);
    expertIdsShape->GetOriginShape().SetDims({numTokens, topK});

    // output 1: weights [numTokens, topK] FP16
    gert::Shape* weightsShape = context->GetOutputShape(1);
    weightsShape->GetOriginShape().SetDims({numTokens, topK});

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MoeRoutingTopK InferDataType
// ============================================================================
static ge::graphStatus MoeRoutingInferDataType(gert::InferDataTypeContext* context)
{
    // expert_ids: INT32
    context->SetOutputDataType(0, ge::DT_INT32);
    // weights: FP16
    context->SetOutputDataType(1, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MoeExpertFFN TilingFunc
// ============================================================================
static ge::graphStatus MoeExpertFFNTilingFunc(gert::TilingContext* context)
{
    OP_LOGI(context->GetNodeName(), "MoeExpertFFN TilingFunc begin.");

    // ---- 1. 获取输入 shape ----
    auto xShape             = context->GetInputShape(0);  // x: [numTokens, hiddenSize]
    auto expertIdsShape     = context->GetInputShape(1);  // expert_ids: [numTokens, topK]
    auto weightsShape       = context->GetInputShape(2);  // weights: [numTokens, topK]
    auto expertWeightsShape = context->GetInputShape(3);  // expert_weights: [numExperts, 3*intermediateSize, hiddenSize]

    if (xShape == nullptr || expertIdsShape == nullptr ||
        weightsShape == nullptr || expertWeightsShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "输入 shape 为空，请检查算子输入是否正确连接。");
        return ge::GRAPH_FAILED;
    }

    const auto& xDims  = xShape->GetOriginShape().GetDims();
    const auto& ewDims = expertWeightsShape->GetOriginShape().GetDims();

    // ---- 2. shape 合法性校验 ----
    if (xDims.size() != 2) {
        OP_LOGE(context->GetNodeName(), "x 需为 2D, 实际 dims=%zu。", xDims.size());
        return ge::GRAPH_FAILED;
    }
    if (ewDims.size() != 3) {
        OP_LOGE(context->GetNodeName(), "expert_weights 需为 3D, 实际 dims=%zu。", ewDims.size());
        return ge::GRAPH_FAILED;
    }

    // ---- 3. 提取参数 ----
    const uint32_t numTokens        = static_cast<uint32_t>(xDims[0]);
    const uint32_t hiddenSize       = static_cast<uint32_t>(xDims[1]);
    const uint32_t numExperts       = static_cast<uint32_t>(ewDims[0]);
    const uint32_t intermediateSize = static_cast<uint32_t>(ewDims[1]) / 3;  // 3*intermediate / 3

    // 从 expert_ids shape 获取 topK
    const auto& eidDims = expertIdsShape->GetOriginShape().GetDims();
    const uint32_t topK = (eidDims.size() >= 2) ? static_cast<uint32_t>(eidDims[1]) : DEFAULT_TOP_K;

    // 从属性读取 intermediateSize（若存在）
    uint32_t interSize = intermediateSize;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrsNumber() > 0) {
        const int64_t* interAttr = attrs->GetAttrPointer<int64_t>(0);
        if (interAttr != nullptr && *interAttr > 0) {
            interSize = static_cast<uint32_t>(*interAttr);
        }
    }

    // ---- 4. 获取平台信息 ----
    uint32_t blockDim = DEFAULT_BLOCK_DIM;
    auto platformInfo = platform_ascendc::PlatformInfoAscendC(context->GetPlatformInfo());
    if (platformInfo.IsSet()) {
        auto aiCoreSpec = platformInfo.GetCoreSpecAic();
        uint32_t coreNum = aiCoreSpec.vectorCoreNum;
        if (coreNum > 0) {
            blockDim = coreNum;
        }
        OP_LOGI(context->GetNodeName(),
                "平台信息: vectorCoreNum=%u, ubSize=%lu Bytes。",
                coreNum, aiCoreSpec.ubSize);
    } else {
        OP_LOGW(context->GetNodeName(), "未获取到平台信息，使用默认参数 blockDim=%u。", blockDim);
    }

    // ---- 5. 分块参数 ----
    uint32_t tileTokens      = FFN_TILE_TOKENS;
    uint32_t tileIntermediate = FFN_TILE_INTERMEDIATE;
    uint32_t tileK           = FFN_TILE_K;

    // 验证 UB (两阶段设计，取 max):
    // Stage A (SwiGLU):
    //   xBuf:        tileTokens * hiddenSize * 2        = 16*2048*2 = 64KB
    //   wTileBuf:    tileK * tileIntermediate * 2       = 128*256*2 = 64KB
    //   projBuf:     tileTokens * tileIntermediate * 4  = 16*256*4  = 16KB (gate FP32, 复用于 up)
    //   interBuf:    tileTokens * tileIntermediate * 2  = 16*256*2  = 8KB  (SwiGLU FP16)
    //   tempBuf:     tileTokens * tileIntermediate * 4  = 16*256*4  = 16KB
    //   总计 ≈ 168KB + tiling ≈ 170KB < 256KB ✓
    // Stage B (Down proj):
    //   interBuf(8KB) + wTileBuf(64KB) + projBuf(16KB) + tempBuf(16KB) ≈ 104KB
    // max(Stage_A, Stage_B) = Stage_A ≈ 170KB ✓

    // 任务分配: 每专家一个任务，内部循环 token 块
    const uint32_t totalTaskNum = numExperts;

    // ---- 6. 填充 TilingData ----
    MoeExpertFFNTilingData tiling;
    tiling.set_numTokens(numTokens);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_intermediateSize(interSize);
    tiling.set_numExperts(numExperts);
    tiling.set_topK(topK);
    tiling.set_tileTokens(tileTokens);
    tiling.set_tileIntermediate(tileIntermediate);
    tiling.set_tileK(tileK);
    tiling.set_blockDim(blockDim);
    tiling.set_totalTaskNum(totalTaskNum);

    // ---- 7. 设置 BlockDim ----
    context->SetBlockDim(blockDim);

    // ---- 8. 序列化 TilingData ----
    if (!tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                              context->GetRawTilingData()->GetCapacity())) {
        OP_LOGE(context->GetNodeName(), "TilingData 序列化失败，缓冲区容量不足。");
        return ge::GRAPH_FAILED;
    }
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // ---- 9. Workspace 设置 ----
    // workspace 用于存储 intermediate 结果 (两阶段设计):
    // 大小 = numExperts * MAX_TILE_BATCHES * tileTokens * intermediateSize * sizeof(half)
    // MAX_TILE_BATCHES = 32 (与 kernel 中一致)
    auto workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes != nullptr) {
        constexpr uint32_t MAX_TILE_BATCHES = 32;
        workspaceSizes[0] = static_cast<size_t>(numExperts) * MAX_TILE_BATCHES *
                             tileTokens * interSize * sizeof(uint16_t);  // half = 2 bytes
    }

    OP_LOGI(context->GetNodeName(),
            "Tiling 完成: numTokens=%u hiddenSize=%u intermediateSize=%u numExperts=%u topK=%u "
            "tileTokens=%u tileIntermediate=%u tileK=%u totalTaskNum=%u blockDim=%u workspace=%zuB",
            numTokens, hiddenSize, interSize, numExperts, topK,
            tileTokens, tileIntermediate, tileK, totalTaskNum, blockDim,
            workspaceSizes ? workspaceSizes[0] : 0);

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MoeExpertFFN InferShape
// ============================================================================
static ge::graphStatus MoeExpertFFNInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape          = context->GetInputShape(0);
    const gert::Shape* expertIdsShape  = context->GetInputShape(1);
    if (xShape == nullptr || expertIdsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& xDims   = xShape->GetOriginShape().GetDims();
    const auto& eidDims = expertIdsShape->GetOriginShape().GetDims();

    if (xDims.size() != 2 || eidDims.size() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int64_t numTokens  = xDims[0];
    const int64_t topK       = eidDims[1];
    const int64_t hiddenSize = xDims[1];

    // output: expert_out [numTokens, topK, hiddenSize]
    gert::Shape* outShape = context->GetOutputShape(0);
    outShape->GetOriginShape().SetDims({numTokens, topK, hiddenSize});

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MoeExpertFFN InferDataType
// ============================================================================
static ge::graphStatus MoeExpertFFNInferDataType(gert::InferDataTypeContext* context)
{
    // expert_out: FP16
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MoeCombine TilingFunc
// ============================================================================
static ge::graphStatus MoeCombineTilingFunc(gert::TilingContext* context)
{
    OP_LOGI(context->GetNodeName(), "MoeCombine TilingFunc begin.");

    // ---- 1. 获取输入 shape ----
    auto expertOutShape = context->GetInputShape(0);  // expert_out: [numTokens, topK, hiddenSize]
    auto expertIdsShape = context->GetInputShape(1);  // expert_ids: [numTokens, topK]
    auto weightsShape   = context->GetInputShape(2);  // weights: [numTokens, topK]

    if (expertOutShape == nullptr || expertIdsShape == nullptr || weightsShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "输入 shape 为空，请检查算子输入是否正确连接。");
        return ge::GRAPH_FAILED;
    }

    const auto& outDims  = expertOutShape->GetOriginShape().GetDims();
    const auto& eidDims  = expertIdsShape->GetOriginShape().GetDims();

    // ---- 2. shape 合法性校验 ----
    if (outDims.size() != 3) {
        OP_LOGE(context->GetNodeName(), "expert_out 需为 3D, 实际 dims=%zu。", outDims.size());
        return ge::GRAPH_FAILED;
    }
    if (eidDims.size() != 2) {
        OP_LOGE(context->GetNodeName(), "expert_ids 需为 2D, 实际 dims=%zu。", eidDims.size());
        return ge::GRAPH_FAILED;
    }

    // ---- 3. 提取参数 ----
    const uint32_t numTokens  = static_cast<uint32_t>(outDims[0]);
    const uint32_t topK       = static_cast<uint32_t>(outDims[1]);
    const uint32_t hiddenSize = static_cast<uint32_t>(outDims[2]);

    // ---- 4. 获取平台信息 ----
    uint32_t blockDim = DEFAULT_BLOCK_DIM;
    auto platformInfo = platform_ascendc::PlatformInfoAscendC(context->GetPlatformInfo());
    if (platformInfo.IsSet()) {
        auto aiCoreSpec = platformInfo.GetCoreSpecAic();
        uint32_t coreNum = aiCoreSpec.vectorCoreNum;
        if (coreNum > 0) {
            blockDim = coreNum;
        }
        OP_LOGI(context->GetNodeName(),
                "平台信息: vectorCoreNum=%u, ubSize=%lu Bytes。",
                coreNum, aiCoreSpec.ubSize);
    } else {
        OP_LOGW(context->GetNodeName(), "未获取到平台信息，使用默认参数 blockDim=%u。", blockDim);
    }

    // ---- 5. 分块参数 ----
    uint32_t tileS = COMBINE_TILE_S;
    if (tileS > numTokens) tileS = numTokens;

    const uint32_t totalTaskNum = (numTokens + tileS - 1) / tileS;

    // ---- 6. 填充 TilingData ----
    MoeCombineTilingData tiling;
    tiling.set_numTokens(numTokens);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_topK(topK);
    tiling.set_tileS(tileS);
    tiling.set_blockDim(blockDim);
    tiling.set_totalTaskNum(totalTaskNum);

    // ---- 7. 设置 BlockDim ----
    context->SetBlockDim(blockDim);

    // ---- 8. 序列化 TilingData ----
    if (!tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                              context->GetRawTilingData()->GetCapacity())) {
        OP_LOGE(context->GetNodeName(), "TilingData 序列化失败，缓冲区容量不足。");
        return ge::GRAPH_FAILED;
    }
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // ---- 9. Workspace 设置 ----
    auto workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes != nullptr) {
        workspaceSizes[0] = 0;
    }

    OP_LOGI(context->GetNodeName(),
            "Tiling 完成: numTokens=%u hiddenSize=%u topK=%u "
            "tileS=%u totalTaskNum=%u blockDim=%u",
            numTokens, hiddenSize, topK,
            tileS, totalTaskNum, blockDim);

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MoeCombine InferShape
// ============================================================================
static ge::graphStatus MoeCombineInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* expertOutShape = context->GetInputShape(0);
    if (expertOutShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& outDims = expertOutShape->GetOriginShape().GetDims();
    if (outDims.size() != 3) {
        return ge::GRAPH_FAILED;
    }

    const int64_t numTokens  = outDims[0];
    const int64_t hiddenSize = outDims[2];

    // output: y [numTokens, hiddenSize]
    gert::Shape* yShape = context->GetOutputShape(0);
    yShape->GetOriginShape().SetDims({numTokens, hiddenSize});

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MoeCombine InferDataType
// ============================================================================
static ge::graphStatus MoeCombineInferDataType(gert::InferDataTypeContext* context)
{
    // y: FP16
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

// ============================================================================
// 算子原型注册
// ============================================================================
namespace ops {

// ============================================================================
// Kernel 1: MoeRoutingTopK - 门控路由 + Top-K 专家选择
// ============================================================================
class MoeRoutingTopK : public OpDef {
public:
    explicit MoeRoutingTopK(const char* name) : OpDef(name)
    {
        // 输入: x [numTokens, hiddenSize] - FP16, ND 排布
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // 输入: w_gate [hiddenSize, numExperts] - FP16, ND 排布
        this->Input("w_gate")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // 输出: expert_ids [numTokens, topK] - INT32, ND 排布
        this->Output("expert_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});

        // 输出: weights [numTokens, topK] - FP16, ND 排布
        this->Output("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // 属性: topK - Top-K 选择数，默认 2
        this->Attr("topK")
            .Int(2);

        // 关联 InferShape / InferDataType / TilingFunc
        this->SetInferShape(optiling::MoeRoutingInferShape);
        this->SetInferDataType(optiling::MoeRoutingInferDataType);
        this->AICore().SetTiling(optiling::MoeRoutingTilingFunc);

        // 注册支持的 AI 处理器型号
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MoeRoutingTopK);

// ============================================================================
// Kernel 2: MoeExpertFFN - 专家 FFN 前向 (GEMM + SwiGLU + GEMM)
// ============================================================================
class MoeExpertFFN : public OpDef {
public:
    explicit MoeExpertFFN(const char* name) : OpDef(name)
    {
        // 输入: x [numTokens, hiddenSize] - FP16, ND 排布
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // 输入: expert_ids [numTokens, topK] - INT32, ND 排布
        this->Input("expert_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});

        // 输入: weights [numTokens, topK] - FP16, ND 排布
        this->Input("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // 输入: expert_weights [numExperts, 3*intermediateSize, hiddenSize] - FP16, ND 排布
        this->Input("expert_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // 输出: expert_out [numTokens, topK, hiddenSize] - FP16, ND 排布
        this->Output("expert_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // 属性: intermediateSize - FFN 中间层维度，默认 8192
        this->Attr("intermediateSize")
            .Int(8192);

        // 关联 InferShape / InferDataType / TilingFunc
        this->SetInferShape(optiling::MoeExpertFFNInferShape);
        this->SetInferDataType(optiling::MoeExpertFFNInferDataType);
        this->AICore().SetTiling(optiling::MoeExpertFFNTilingFunc);

        // 注册支持的 AI 处理器型号
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MoeExpertFFN);

// ============================================================================
// Kernel 3: MoeCombine - 加权合并专家输出
// ============================================================================
class MoeCombine : public OpDef {
public:
    explicit MoeCombine(const char* name) : OpDef(name)
    {
        // 输入: expert_out [numTokens, topK, hiddenSize] - FP16, ND 排布
        this->Input("expert_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // 输入: expert_ids [numTokens, topK] - INT32, ND 排布
        this->Input("expert_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});

        // 输入: weights [numTokens, topK] - FP16, ND 排布
        this->Input("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // 输出: y [numTokens, hiddenSize] - FP16, ND 排布
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // 属性: topK - Top-K 选择数，默认 2
        this->Attr("topK")
            .Int(2);

        // 关联 InferShape / InferDataType / TilingFunc
        this->SetInferShape(optiling::MoeCombineInferShape);
        this->SetInferDataType(optiling::MoeCombineInferDataType);
        this->AICore().SetTiling(optiling::MoeCombineTilingFunc);

        // 注册支持的 AI 处理器型号
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MoeCombine);

}  // namespace ops
