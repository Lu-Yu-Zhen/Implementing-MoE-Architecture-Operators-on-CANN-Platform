# ============================================================================
# 测试用例 CMakeLists.txt
# 编译单算子调用测试程序（需先完成算子部署: ./build_out/custom_opp_*.run）
# ============================================================================

cmake_minimum_required(VERSION 3.16)

project(moe_test)

set(CMAKE_CXX_STANDARD 17)

# CANN 路径
set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_CANN_PACKAGE_PATH})
if(NOT ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/ascend-toolkit/latest)
endif()

# 头文件搜索路径
include_directories(
    ${ASCEND_CANN_PACKAGE_PATH}/runtime/include
    ${ASCEND_CANN_PACKAGE_PATH}/atc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../op_host
)

# 链接库
link_directories(
    ${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/atc/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/tools/opp/vendors/customize/op_api/lib
)

add_executable(test_moe test_moe.cpp)

target_link_libraries(test_moe
    ascendcl
    acl_op_compiler
    nnopbase
    cust_opapi
    stdc++
)
