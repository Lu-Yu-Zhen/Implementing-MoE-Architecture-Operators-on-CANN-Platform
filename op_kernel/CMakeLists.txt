# ============================================================================
# Kernel 侧 CMakeLists.txt
# 编译算子核函数为 .o 二进制（ccec 工具链，由 CANN 注入）
# ============================================================================

# 算子 Kernel 源文件（每个算子独立源文件）
set(MOE_ROUTING_SRC    ${CMAKE_CURRENT_SOURCE_DIR}/moe_routing.cpp)
set(MOE_EXPERT_FFN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/moe_expert_ffn.cpp)
set(MOE_COMBINE_SRC    ${CMAKE_CURRENT_SOURCE_DIR}/moe_combine.cpp)

# 算子信息配置文件
set(OP_INFO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/op_info.json)

# [关键] 调用 CANN 提供的 add_ops_compile_options
# 为每个算子分别配置 Kernel 编译选项
# OP_TYPE: 算子类型名（需与 op_host 注册的算子名一致）
# SRCS:    kernel 源文件列表

# Kernel 1: MoeRoutingTopK - 门控路由 + Top-K 选择
add_ops_compile_options(
    OP_TYPE MoeRoutingTopK
    SRCS ${MOE_ROUTING_SRC}
)

# Kernel 2: MoeExpertFFN - 专家 FFN 前向（GEMM + SwiGLU）
add_ops_compile_options(
    OP_TYPE MoeExpertFFN
    SRCS ${MOE_EXPERT_FFN_SRC}
)

# Kernel 3: MoeCombine - 加权合并专家输出
add_ops_compile_options(
    OP_TYPE MoeCombine
    SRCS ${MOE_COMBINE_SRC}
)

# [关键] 调用 CANN 提供的 add_ops_kernel
# 将编译好的 .o 文件打包为目标库，生成算子二进制与信息库
add_ops_kernel(
    TARGET moe_kernel
    OPS_INFO ${OP_INFO_FILE}
)
