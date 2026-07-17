# ============================================================================
# CANN 公共 CMake 工具函数库（占位实现）
# ============================================================================
# 注意：本文件为占位实现。实际使用时，CANN 工具包会在以下路径提供官方版本：
#   ${ASCEND_CANN_PACKAGE_PATH}/compiler/cmake/util/utils.cmake
#
# 正式部署时应将 CANN 提供的 utils.cmake 拷贝覆盖本文件，或通过软链接指向
# 官方文件，以获取完整的 add_ops_compile_options / add_ops_kernel 等函数实现。
# ============================================================================

# 引入 CANN 官方 utils.cmake（若存在则覆盖下方占位实现）
if(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/compiler/cmake/util/utils.cmake")
    include("${ASCEND_CANN_PACKAGE_PATH}/compiler/cmake/util/utils.cmake")
    return()
endif()

# --------------------------- 占位函数定义 ---------------------------
# 当无法找到官方 utils.cmake 时，提供以下占位实现以保证工程可被 CMake 解析
# 实际编译时需安装 CANN 工具包并由其提供真实实现

include_guard(GLOBAL)

# add_ops_compile_options: 配置算子 Kernel 编译选项
# 作用：自动检测环境，调用 ccec 编译器，注入 Ascend C 所需宏与架构参数
function(add_ops_compile_options)
    set(options ALL_OPS)
    set(oneValueArgs OP_TYPE)
    set(multiValueArgs SRCS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_OP_TYPE)
        message(FATAL_ERROR "add_ops_compile_options 缺少 OP_TYPE 参数")
    endif()

    # 占位：实际由 CANN utils.cmake 实现 ccec 编译逻辑
    message(STATUS "[Placeholder] add_ops_compile_options: OP_TYPE=${ARG_OP_TYPE}")
    message(STATUS "[Placeholder]   SRCS=${ARG_SRCS}")
    message(STATUS "[Placeholder] 请确保已安装 CANN 工具包并正确配置 ASCEND_CANN_PACKAGE_PATH")
endfunction()

# add_ops_kernel: 将编译产物打包为目标库
function(add_ops_kernel)
    set(oneValueArgs TARGET)
    set(multiValueArgs OPS_INFO)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_TARGET)
        message(FATAL_ERROR "add_ops_kernel 缺少 TARGET 参数")
    endif()

    message(STATUS "[Placeholder] add_ops_kernel: TARGET=${ARG_TARGET}")
    message(STATUS "[Placeholder]   OPS_INFO=${ARG_OPS_INFO}")
endfunction()

# add_ops_info: 生成算子信息库
function(add_ops_info)
    message(STATUS "[Placeholder] add_ops_info called")
endfunction()
