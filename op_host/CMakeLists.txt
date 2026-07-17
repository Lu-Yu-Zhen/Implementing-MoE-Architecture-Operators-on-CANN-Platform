# ============================================================================
# Host 侧 CMakeLists.txt
# 编译算子原型库、Tiling 库（g++ 工具链）
# ============================================================================

# 算子 Host 侧源文件
file(GLOB HOST_SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# 算子 Host 侧头文件（供其他模块引用）
set(HOST_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# 引入 CANN Host 侧头文件
include_directories(
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/tikcfp
    ${HOST_INC_DIR}
)

# 编译算子原型与 Tiling 实现为动态库
# 实际由 CANN utils.cmake 的 add_ops_info / add_ops_proto 等函数处理
# 此处声明源文件依赖关系，具体编译逻辑由 CANN 工具链接管
if(COMMAND add_ops_info)
    add_ops_info(
        OPS_INFO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/op_info.json
        SRCS ${HOST_SRC_FILES}
    )
endif()

# 供上层调试用的本地库目标（开发阶段可链接验证）
if(NOT COMMAND add_ops_info)
    add_library(moe_host SHARED ${HOST_SRC_FILES})
    target_compile_features(moe_host PRIVATE cxx_std_17)
    target_compile_options(moe_host PRIVATE -O2 -fPIC -Wall -Wno-unused-parameter)
endif()
