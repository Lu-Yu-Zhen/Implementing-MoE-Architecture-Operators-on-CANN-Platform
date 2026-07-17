#!/bin/bash
# ============================================================================
# MoE 算子编译入口脚本
# 用法:
#   ./build.sh                 # 默认编译 (ascend910b)
#   ./build.sh ascend910b      # 指定 SoC 版本
#   ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest ./build.sh
# ============================================================================

set -e

# 默认 SoC 版本：Ascend 910B（Atlas A2 训练卡）
SOC_VERSION=${1:-ascend910b}

# CANN 安装路径（可由环境变量覆盖）
export ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/ascend-toolkit/latest}

if [ ! -d "${ASCEND_CANN_PACKAGE_PATH}" ]; then
    echo "[ERROR] ASCEND_CANN_PACKAGE_PATH 不存在: ${ASCEND_CANN_PACKAGE_PATH}"
    echo "        请安装 CANN 工具包或通过环境变量指定正确路径"
    exit 1
fi

echo "============================================================"
echo " MoE 算子编译"
echo "   SoC Version            : ${SOC_VERSION}"
echo "   CANN Package Path      : ${ASCEND_CANN_PACKAGE_PATH}"
echo "============================================================"

# 清理旧构建产物
rm -rf build build_out

# CMake 配置阶段
cmake -S . -B build \
    -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH} \
    -DASCEND_SOC_VERSION=${SOC_VERSION} \
    -DCMAKE_INSTALL_PREFIX=$(pwd)/build_out

# CMake 编译并安装，生成 custom_opp_<os>_<arch>.run 安装包
cmake --build build -j$(nproc) --target install

echo ""
echo "============================================================"
echo " 编译成功！"
echo " 产物目录: $(pwd)/build_out"
echo " 安装包  : $(ls build_out/*.run 2>/dev/null)"
echo " 部署命令: ./build_out/custom_opp_*.run"
echo "============================================================"
