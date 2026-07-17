#ifndef MOE_TILING_H
#define MOE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

// ============================================================================
// MoE (Mixture of Experts) TilingData 结构定义
// ----------------------------------------------------------------------------
// MoE 采用三阶段流水线：
//   Kernel 1 (MoeRoutingTopK) : 门控路由，计算 logits = x @ w_gate，取 Top-K
//   Kernel 2 (MoeExpertFFN)   : 专家 FFN，对每个专家执行 GEMM + SwiGLU + GEMM
//   Kernel 3 (MoeCombine)     : 加权合并，y = Σ weight_k * expert_out_k
// 目标硬件: Ascend 910B (24 AI Cores, ~256KB UB/core)
// ============================================================================

// ============================================================================
// MoeRoutingTilingData (Kernel 1: MoeRoutingTopK)
// ----------------------------------------------------------------------------
// W_gate 常驻 UB: hiddenSize * numExperts * 2 = 2048*16*2 = 64KB
// x_tile: tileS * hiddenSize * 2 bytes
// logits: tileS * numExperts * 4 bytes (FP32 累加)
// UB 约束: 64KB + tileS*2048*2 + tileS*16*4 <= 256KB
//          tileS <= (256-64)*1024 / (2048*2 + 16*4) ≈ 45 → 取 32
// ============================================================================
BEGIN_TILING_DATA_DEF(MoeRoutingTilingData)

    // ---------------- 输入形状参数 ----------------
    TILING_DATA_FIELD_DEF(uint32_t, numTokens);      // token 总数
    TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);     // 隐藏层维度 (2048)
    TILING_DATA_FIELD_DEF(uint32_t, numExperts);     // 专家数 (16)
    TILING_DATA_FIELD_DEF(uint32_t, topK);           // Top-K 选择数 (2)

    // ---------------- 分块参数 ----------------
    TILING_DATA_FIELD_DEF(uint32_t, tileS);          // 每核每次处理的 token 数 (32)

    // ---------------- 硬件调度参数 ----------------
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);       // 使用的 AI Core 核数
    TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum);   // 总任务数 = ceil(numTokens / tileS)
    TILING_DATA_FIELD_DEF(uint32_t, coreTaskNum);    // 每核任务数 = ceil(totalTaskNum / blockDim)

END_TILING_DATA_DEF;

// 将 TilingData 结构绑定到 MoeRoutingTopK 算子
REGISTER_TILING_DATA_CLASS(MoeRoutingTopK, MoeRoutingTilingData)

// ============================================================================
// MoeExpertFFNTilingData (Kernel 2: MoeExpertFFN - 最关键)
// ----------------------------------------------------------------------------
// 两阶段设计 (tileTokens=16)，两阶段不同时执行，取 max:
// Stage A (SwiGLU):
//   xBuf:        tileTokens * hiddenSize * 2        = 16*2048*2 = 64KB
//   wTileBuf:    tileK * tileIntermediate * 2       = 128*256*2 = 64KB
//   projBuf:     tileTokens * tileIntermediate * 4  = 16*256*4  = 16KB (gate FP32, 复用于 up)
//   interBuf:    tileTokens * tileIntermediate * 2  = 16*256*2  = 8KB  (SwiGLU FP16)
//   tempBuf:     tileTokens * tileIntermediate * 4  = 16*256*4  = 16KB
//   总计 ≈ 168KB + tiling ≈ 170KB < 256KB
// Stage B (Down proj):
//   interBuf(8KB) + wTileBuf(64KB) + projBuf(16KB) + tempBuf(16KB) ≈ 104KB
// max(Stage_A, Stage_B) = Stage_A ≈ 170KB ✓
// workspace 大小: numExperts * 32 * tileTokens * intermediateSize * sizeof(half)
// 任务分配: totalTaskNum = numExperts，每专家一个任务，内部循环 token 块
// ============================================================================
BEGIN_TILING_DATA_DEF(MoeExpertFFNTilingData)

    // ---------------- 输入形状参数 ----------------
    TILING_DATA_FIELD_DEF(uint32_t, numTokens);         // token 总数
    TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);        // 隐藏层维度 (2048)
    TILING_DATA_FIELD_DEF(uint32_t, intermediateSize);  // FFN 中间层维度 (8192)
    TILING_DATA_FIELD_DEF(uint32_t, numExperts);        // 专家数 (16)
    TILING_DATA_FIELD_DEF(uint32_t, topK);              // Top-K 选择数 (2)

    // ---------------- 分块参数 ----------------
    TILING_DATA_FIELD_DEF(uint32_t, tileTokens);       // 每专家每次处理的 token 块大小 (16)
    TILING_DATA_FIELD_DEF(uint32_t, tileIntermediate); // intermediate 维分块大小 (256)
    TILING_DATA_FIELD_DEF(uint32_t, tileK);            // hidden 维分块大小，MatmulImpl 的 K 维 tile (128)

    // ---------------- 硬件调度参数 ----------------
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);         // 使用的 AI Core 核数
    TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum);     // 总任务数 = numExperts

END_TILING_DATA_DEF;

// 将 TilingData 结构绑定到 MoeExpertFFN 算子
REGISTER_TILING_DATA_CLASS(MoeExpertFFN, MoeExpertFFNTilingData)

// ============================================================================
// MoeCombineTilingData (Kernel 3: MoeCombine)
// ----------------------------------------------------------------------------
// 简单按 token 分块，每 token 累加 topK 个专家输出:
//   expert_out_tile: tileS * topK * hiddenSize * 2
//   y_tile: tileS * hiddenSize * 2
//   UB 约束: tileS * (topK + 1) * hiddenSize * 2 <= 256KB
//            tileS <= 256*1024 / (3 * 2048 * 2) ≈ 21 → 取 64 (实际可调整)
// ============================================================================
BEGIN_TILING_DATA_DEF(MoeCombineTilingData)

    // ---------------- 输入形状参数 ----------------
    TILING_DATA_FIELD_DEF(uint32_t, numTokens);      // token 总数
    TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);     // 隐藏层维度 (2048)
    TILING_DATA_FIELD_DEF(uint32_t, topK);           // Top-K 选择数 (2)

    // ---------------- 分块参数 ----------------
    TILING_DATA_FIELD_DEF(uint32_t, tileS);          // 每次处理的 token 数 (64)

    // ---------------- 硬件调度参数 ----------------
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);       // 使用的 AI Core 核数
    TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum);   // 总任务数 = ceil(numTokens / tileS)

END_TILING_DATA_DEF;

// 将 TilingData 结构绑定到 MoeCombine 算子
REGISTER_TILING_DATA_CLASS(MoeCombine, MoeCombineTilingData)

}  // namespace optiling

#endif  // MOE_TILING_H
