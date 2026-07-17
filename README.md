# Implementing MoE Architecture Operators on CANN Platform

## 在 CANN 平台上实现 MoE 架构算子

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English

### Overview

This repository implements **Mixture of Experts (MoE)** architecture operators optimized for the **Huawei Ascend CANN (Compute Architecture for Neural Networks)** platform. MoE is a powerful neural network architecture paradigm that enables scaling model capacity while keeping inference costs manageable through sparse activation of expert sub-networks.

### Key Features

- **Multiple Routing Strategies**
  - Top-K Router: Classic sparse gating (from GShard / Switch Transformer)
  - Switch Router: Top-1 expert selection per token (Switch Transformer style)
  - Configurable Gating Network: Linear or MLP-based gating

- **Expert Network Variants**
  - Standard Feed-Forward Expert (FFN)
  - Shared Expert (DeepSeekMoE style)
  - Expert Groups for parallel dispatch

- **CANN-Optimized Operators**
  - `CannSwitchRouter`: Fused routing on Ascend NPU
  - `CannAllToAll`: HCCL-optimized all-to-all for expert parallelism
  - `CannFusedMoE`: Single-kernel fused MoE computation

- **Load Balancing**
  - Auxiliary load balancing loss
  - Expert utilization metrics
  - Z-loss for gating stability

- **Device Agnostic**
  - Ascend NPU (via torch-npu)
  - CUDA GPU (fallback)
  - CPU (fallback)

### Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── moe/                    # MoE core modules
│   │   ├── __init__.py
│   │   ├── router.py           # Router & gating implementations
│   │   ├── experts.py          # Expert network implementations
│   │   ├── layers.py           # MoE layer compositions
│   │   ├── load_balance.py     # Load balancing utilities
│   │   └── utils.py            # Device utilities
│   ├── cann_ops/               # CANN-specific operators
│   │   ├── __init__.py
│   │   ├── switch_router.py    # CANN switch router
│   │   ├── all_to_all.py       # CANN all-to-all communication
│   │   └── fused_moe.py        # CANN fused MoE kernel
│   └── examples/               # Usage examples
│       ├── __init__.py
│       ├── basic_moe.py        # Basic MoE demo
│       └── advanced_moe.py     # Advanced MoE with CANN ops
├── tests/
│   ├── __init__.py
│   ├── test_router.py          # Router tests
│   ├── test_experts.py         # Expert tests
│   └── test_moe_layer.py       # MoE layer tests
├── setup.py
├── requirements.txt
├── LICENSE
└── README.md
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Implementing-MoE-Architecture-Operators-on-CANN-Platform

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Quick Start

```python
import torch
from moe import MoELayer, LoadBalanceLoss
from moe.utils import get_device

# Auto-detect device (NPU > CUDA > CPU)
device = get_device()

# Create an MoE layer with 8 experts, top-2 routing
moe = MoELayer(
    hidden_size=256,
    num_experts=8,
    top_k=2,
    intermediate_size=1024,
    capacity_factor=1.25,
).to(device)

# Forward pass
x = torch.randn(2, 16, 256, device=device)
output, aux_loss = moe(x)

print(f"Output shape: {output.shape}")
print(f"Aux loss: {aux_loss.item():.6f}")
```

### CANN Fused MoE

For maximum performance on Ascend NPU, use the fused CANN operator:

```python
from cann_ops import CannFusedMoE

# Single-kernel fused MoE computation
fused_moe = CannFusedMoE(
    hidden_size=256,
    num_experts=8,
    top_k=2,
).to("npu")

output, aux_loss = fused_moe(x)
```

### Running Tests

```bash
pytest tests/ -v
```

### Running Examples

```bash
# Basic MoE demo
python src/examples/basic_moe.py

# Advanced MoE with CANN ops
python src/examples/advanced_moe.py
```

### References

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)
- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)

### License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

<a name="chinese"></a>
## 中文

### 概述

本仓库在 **华为昇腾 CANN（Compute Architecture for Neural Networks）** 平台上实现了 **MoE（Mixture of Experts，混合专家）** 架构算子。MoE 是一种强大的神经网络架构范式，通过稀疏激活专家子网络，在控制推理成本的同时大幅扩展模型容量。

### 核心特性

- **多种路由策略**
  - Top-K 路由：经典稀疏门控（源自 GShard / Switch Transformer）
  - Switch 路由：每个 token 选择 Top-1 专家（Switch Transformer 风格）
  - 可配置门控网络：线性或 MLP 门控

- **专家网络变体**
  - 标准前馈专家（FFN）
  - 共享专家（DeepSeekMoE 风格）
  - 专家组并行调度

- **CANN 优化算子**
  - `CannSwitchRouter`：昇腾 NPU 融合路由
  - `CannAllToAll`：基于 HCCL 的专家并行全交换
  - `CannFusedMoE`：单 Kernel 融合 MoE 计算

- **负载均衡**
  - 辅助负载均衡损失
  - 专家利用率指标
  - 门控稳定性 Z-loss

- **设备无关**
  - 昇腾 NPU（通过 torch-npu）
  - CUDA GPU（回退）
  - CPU（回退）

### 项目结构

```
.
├── src/
│   ├── __init__.py
│   ├── moe/                    # MoE 核心模块
│   │   ├── __init__.py
│   │   ├── router.py           # 路由器与门控实现
│   │   ├── experts.py          # 专家网络实现
│   │   ├── layers.py           # MoE 层组合
│   │   ├── load_balance.py     # 负载均衡工具
│   │   └── utils.py            # 设备工具
│   ├── cann_ops/               # CANN 专用算子
│   │   ├── __init__.py
│   │   ├── switch_router.py    # CANN Switch 路由
│   │   ├── all_to_all.py       # CANN 全交换通信
│   │   └── fused_moe.py        # CANN 融合 MoE Kernel
│   └── examples/               # 使用示例
│       ├── __init__.py
│       ├── basic_moe.py        # 基础 MoE 示例
│       └── advanced_moe.py     # 高级 MoE + CANN 算子
├── tests/
│   ├── __init__.py
│   ├── test_router.py          # 路由器测试
│   ├── test_experts.py         # 专家测试
│   └── test_moe_layer.py       # MoE 层测试
├── setup.py
├── requirements.txt
├── LICENSE
└── README.md
```

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd Implementing-MoE-Architecture-Operators-on-CANN-Platform

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

### 快速开始

```python
import torch
from moe import MoELayer, LoadBalanceLoss
from moe.utils import get_device

# 自动检测设备（NPU > CUDA > CPU）
device = get_device()

# 创建 8 专家、Top-2 路由的 MoE 层
moe = MoELayer(
    hidden_size=256,
    num_experts=8,
    top_k=2,
    intermediate_size=1024,
    capacity_factor=1.25,
).to(device)

# 前向传播
x = torch.randn(2, 16, 256, device=device)
output, aux_loss = moe(x)

print(f"输出形状: {output.shape}")
print(f"辅助损失: {aux_loss.item():.6f}")
```

### CANN 融合 MoE

在昇腾 NPU 上获得最佳性能，使用 CANN 融合算子：

```python
from cann_ops import CannFusedMoE

# 单 Kernel 融合 MoE 计算
fused_moe = CannFusedMoE(
    hidden_size=256,
    num_experts=8,
    top_k=2,
).to("npu")

output, aux_loss = fused_moe(x)
```

### 运行测试

```bash
pytest tests/ -v
```

### 运行示例

```bash
# 基础 MoE 示例
python src/examples/basic_moe.py

# 高级 MoE + CANN 算子
python src/examples/advanced_moe.py
```

### 参考文献

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)
- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)

### 许可证

本项目基于 Apache License 2.0 许可 - 详见 [LICENSE](LICENSE) 文件。