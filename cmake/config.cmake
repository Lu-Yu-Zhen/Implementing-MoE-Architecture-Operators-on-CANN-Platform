# ============================================================================
# CMake 全局配置文件
# 设置算子工程相关的编译选项与目标 SoC 列表
# ============================================================================

# 支持的目标 AI 处理器型号（按算力代际）
# ascend910b : Atlas A2 训练卡，UB 约 256KB/core，24 AI Cores
set(SOC_VERSION_LIST
    "ascend910b"
    CACHE STRING "Supported Ascend SoC versions")

# 默认 SoC 版本（可通过 -DASCEND_SOC_VERSION 覆盖）
if(NOT DEFINED ASCEND_SOC_VERSION)
    set(ASCEND_SOC_VERSION "ascend910b" CACHE STRING "Target SoC version")
endif()

# Host 侧编译选项（g++ 工具链）
set(HOST_CXX_FLAGS
    -std=c++17
    -O2
    -fPIC
    -Wall
    -Wno-unused-parameter
    CACHE STRING "Host side compiler flags")

# Kernel 侧编译选项（ccec 工具链，由 add_ops_compile_options 注入）
set(KERNEL_CXX_FLAGS
    -O2
    CACHE STRING "Kernel side compiler flags")

message(STATUS "[MoE] Target SoC: ${ASCEND_SOC_VERSION}")
message(STATUS "[MoE] CANN Path : ${ASCEND_CANN_PACKAGE_PATH}")
