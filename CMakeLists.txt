# ============================================================================
# MoE 算子工程根 CMakeLists.txt
# 标准 CANN 算子工程结构
# ============================================================================

cmake_minimum_required(VERSION 3.16)

# 加载 CANN 工具链配置
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_CANN_PACKAGE_PATH} CACHE PATH "CANN package path")

if(NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}")
    message(FATAL_ERROR "ASCEND_CANN_PACKAGE_PATH not found: ${ASCEND_CANN_PACKAGE_PATH}")
endif()

# 项目级配置
project(MoE)

# 加载全局配置
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/util/utils.cmake)

# 子目录
add_subdirectory(op_kernel)
add_subdirectory(op_host)
add_subdirectory(framework)

# 可选测试构建
option(BUILD_TESTS "Build test programs" ON)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
