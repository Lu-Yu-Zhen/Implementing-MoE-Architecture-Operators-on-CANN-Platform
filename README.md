# 基于 CANN 平台的 MoE 架构算子实现

> 在华为昇腾 Ascend 910B AI 处理器上，使用 Ascend C 算子编程模型从零实现 Mixture of Experts（MoE）推理算子，覆盖门控路由、专家 FFN、加权合并三阶段流水线。

## 项目简介

Mixture of Experts（MoE）是当前大语言模型（如 DeepSeek-MoE、Mixtral、GPT-4）中广泛采用的稀疏激活架构，其核心思想是为每个 token 动态选择少量专家网络进行计算，从而在不显著增加推理计算量的前提下扩展模型参数规模。

本项目在 CANN（Compute Architecture for Neural Networks）平台上，使用 Ascend C 算子编程模型，针对 Ascend 910B 硬件特性（24 AI Cores、256KB UB/core、Cube/Vector 双引擎）从零实现了完整的 MoE 推理算子流水线，包含以下三个自定义算子：

| 算子 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `MoeRoutingTopK` | 门控路由 + Top-K 专家选择 | `x`, `w_gate` | `expert_ids`, `weights` |
| `MoeExpertFFN` | 专家 FFN 前向（GEMM + SwiGLU + GEMM） | `x`, `expert_ids`, `weights`, `expert_weights` | `expert_out` |
| `MoeCombine` | 加权合并专家输出 | `expert_out`, `expert_ids`, `weights` | `y` |

整体数据流：

```
x ──► [MoeRoutingTopK] ──► expert_ids, weights
 │                              │
 ├──► [MoeExpertFFN] ◄──────────┘
 │        │
 │        ▼
 │     expert_out
 │        │
 └──► [MoeCombine] ◄─────────── weights
          │
          ▼
          y
```

## 目标硬件

| 特性 | 规格 |
|------|------|
| AI 处理器 | Ascend 910B（Atlas A2 训练卡） |
| AI Core 数量 | 24 |
| Unified Buffer (UB) | ~256 KB / core |
| 计算引擎 | Cube（矩阵乘）+ Vector（向量化） |
| 数据类型 | FP16 输入/输出，FP32 GEMM 累加 |
| SoC 版本 | `ascend910b` |

## 目录结构

```
.
├── CMakeLists.txt              # 工程根 CMake 配置
├── CMakePresets.json           # CMake 预设（SoC 版本、CANN 路径）
├── build.sh                    # NPU 算子编译脚本
├── build_cpu_sim.sh            # CPU 仿真编译脚本（无需 CANN 环境）
├── cmake/
│   ├── config.cmake            # 全局编译选项与目标 SoC 列表
│   └── util/
│       └── utils.cmake         # CANN 工具函数（占位实现，部署时由官方覆盖）
├── framework/
│   └── CMakeLists.txt          # 框架适配插件目录（由 msopgen 生成）
├── op_host/                    # Host 侧实现
│   ├── CMakeLists.txt
│   ├── moe_ops.cpp             # 算子原型注册、InferShape、InferDataType、TilingFunc
│   └── moe_tiling.h            # TilingData 结构定义
├── op_kernel/                  # Kernel 侧实现（Device 代码）
│   ├── CMakeLists.txt
│   ├── moe_routing.cpp         # Kernel 1: 门控路由 + Top-K
│   ├── moe_expert_ffn.cpp      # Kernel 2: 专家 FFN（GEMM + SwiGLU）
│   ├── moe_combine.cpp         # Kernel 3: 加权合并
│   └── op_info.json            # 算子信息库配置
└── tests/
    ├── CMakeLists.txt
    ├── test_moe.cpp            # NPU 端到端精度/性能测试
    └── moe_cpu_sim.cpp         # CPU 仿真测试（无需 CANN，验证算法正确性）
```

## 算子详解

### Kernel 1: MoeRoutingTopK — 门控路由 + Top-K 选择

**功能**：对每个 token 计算门控分数，并通过 Top-K 选择最相关的若干专家。

**算法**：
1. `logits = x @ w_gate`（Cube 矩阵乘，FP32 累加）
2. 逐行 Softmax（数值稳定版：减最大值 → exp → 归一化）
3. Top-K 选择（两次扫描找最大/次大专家）
4. Top-K 权重归一化（使选中的 K 个权重之和为 1）
5. Cast FP32 → FP16 输出

**UB 预算**（`tileS=32`，`H=2048`，`E=16`）：

| 缓冲区 | 大小 |
|--------|------|
| `xBuf`（x 块） | 32 × 2048 × 2 = 128 KB |
| `wGateBuf`（常驻） | 2048 × 16 × 2 = 64 KB |
| `logitsBuf`（GEMM 输出） | 32 × 16 × 4 = 2 KB |
| 临时缓冲 | ≈ 16 KB |
| **合计** | **≈ 210 KB < 256 KB** ✓ |

**多核调度**：按 `tileS` 分块，24 核轮询处理任务队列。

### Kernel 2: MoeExpertFFN — 专家 FFN 前向（性能核心）

**功能**：对每个被选中的专家执行 SwiGLU FFN 前向计算。

**SwiGLU FFN 公式**：
```
gate = x @ W_gate^T          [tileTokens, intermediateSize]
up   = x @ W_up^T            [tileTokens, intermediateSize]
hidden = SiLU(gate) * up     [tileTokens, intermediateSize]   ← SwiGLU 激活
out  = hidden @ W_down       [tileTokens, hiddenSize]
```

其中 `SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))`。

**两阶段流水线设计**（取 max，UB 复用）：

| 阶段 | 使用缓冲区 | 合计 |
|------|-----------|------|
| Stage A (SwiGLU) | xBuf(64KB) + wTileBuf(64KB) + projBuf(16KB) + interBuf(8KB) + tempBuf(16KB) | ≈ 170 KB |
| Stage B (Down) | interBuf(8KB) + wTileBuf(64KB) + projBuf(16KB) + tempBuf(16KB) | ≈ 104 KB |
| **max(A, B)** | | **≈ 170 KB < 256 KB** ✓ |

**关键优化**：
- **K-blocking**：沿 `hiddenSize` 维分块（`tileK=128`），MatmulImpl 单次 Mmad 后累加
- **workspace 中转**：Stage A 的 SwiGLU 中间结果写入 GM workspace，Stage B 读回做 Down 投影，避免 UB 溢出
- **专家并行**：每专家一个任务（`totalTaskNum = numExperts`），多核轮询
- **权重布局**：`expert_weights[numExperts, 3*intermediateSize, hiddenSize]`，三段式存储（gate / up / down）

### Kernel 3: MoeCombine — 加权合并

**功能**：将 Top-K 个专家的输出按路由权重加权求和，得到最终输出。

**公式**：
```
y[token, d] = Σ_k  weights[token, k] * expert_out[token, k, d]
```

**优化**：直接在 FP16 下做 `Muls` 和 `Add`，省掉 Cast 开销（合并阶段精度要求不高）。UB 占用仅 ~12KB，远低于 256KB 上限。

## 编译与运行

### 环境要求

- CANN 工具包（>= 7.0）
- CMake >= 3.16
- GCC >= 7.5（Host 侧）或 ccec（Kernel 侧，由 CANN 提供）
- 目标硬件：Ascend 910B（Atlas A2 训练卡）

### 方式一：NPU 算子编译（需 CANN 环境）

```bash
# 设置 CANN 路径（如未设置）
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest

# 默认编译（ascend910b）
./build.sh

# 指定 SoC 版本
./build.sh ascend910b

# 编译产物位于 build_out/，部署：
./build_out/custom_opp_*.run
```

### 方式二：CPU 仿真验证（无需 CANN）

```bash
# 仅需 g++，验证三阶段流水线算法正确性
./build_cpu_sim.sh
```

CPU 仿真测试包含：
- **Tiling 参数验证**：检查分块参数是否满足 UB 约束
- **UB 容量验证**：确保所有缓冲区总和不超 256KB
- **精度对比**：分块实现 vs Naive 参考实现（余弦相似度 > 0.999）
- **性能计时**：Naive vs Tiled 耗时对比

### 方式三：CMake Presets

```bash
cmake --preset default
cmake --build --preset default
```

## 测试

### NPU 端到端测试

```bash
# 需先完成算子部署
cd tests
mkdir build && cd build
cmake .. && make
./test_moe
```

测试用例：
- **小规模**：T=8, H=64, E=4, K=2, I=256
- **目标规模**：T=128, H=2048, E=16, K=2, I=8192

精度阈值：
- 最大绝对误差 < 0.05
- 平均绝对误差 < 0.01
- 余弦相似度 > 0.99

### CPU 仿真测试

```bash
g++ -O2 -std=c++17 tests/moe_cpu_sim.cpp -o moe_cpu_sim -lm
./moe_cpu_sim
```

## 关键技术点

1. **Ascend C 编程模型**：使用 `__aicore__`、`__gm__`、`TPipe`、`TBuf`、`GlobalTensor` 等 CANN 特有语法
2. **MatmulImpl Cube 矩阵乘**：通过 `MatmulType` 模板参数控制数据排布（ND 格式、转置标志）
3. **手动构造 TCubeTiling**：参照 FlashAttention 实现，绕过自动 Tiling 机制以获得更精确的控制
4. **UB 双缓冲与分块**：`BUFFER_NUM=2` 掩盖搬运延迟，`tileS`/`tileTokens`/`tileIntermediate`/`tileK` 多级分块
5. **workspace 两阶段设计**：Stage A 结果写 GM，Stage B 读回继续计算，突破 UB 容量限制
6. **多核轮询调度**：`taskId = coreId; taskId < totalTaskNum; taskId += coreNum` 模式均衡负载

## 默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `numExperts` | 16 | 专家数量 |
| `topK` | 2 | 每 token 选择的专家数 |
| `hiddenSize` | 2048 | 隐藏层维度 |
| `intermediateSize` | 8192 | FFN 中间层维度 |
| `blockDim` | 24 | 使用的 AI Core 数（910B） |
| `tileS` (Routing) | 32 | 路由阶段每核每次处理的 token 数 |
| `tileTokens` (FFN) | 16 | FFN 阶段每专家每次处理的 token 数 |
| `tileIntermediate` | 256 | FFN 中间维度分块大小 |
| `tileK` | 128 | FFN K 维分块大小 |

## 许可证

Apache License 2.0，详见 [LICENSE](LICENSE)。
