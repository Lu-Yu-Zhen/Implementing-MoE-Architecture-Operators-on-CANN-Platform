#!/bin/bash
# ============================================================================
# MoE CPU 仿真编译脚本
# 纯 g++ 编译，不依赖 CANN 环境
# ============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build_cpu_sim"
mkdir -p "${BUILD_DIR}"

echo "=========================================="
echo " MoE CPU 仿真编译"
echo "=========================================="
echo "项目目录: ${PROJECT_DIR}"
echo "构建目录: ${BUILD_DIR}"
echo ""

g++ \
    -O2 -std=c++17 -g \
    "${PROJECT_DIR}/tests/moe_cpu_sim.cpp" \
    -o "${BUILD_DIR}/moe_cpu_sim" \
    -lm

echo ""
echo "编译完成，开始运行..."
echo ""
"${BUILD_DIR}/moe_cpu_sim"
