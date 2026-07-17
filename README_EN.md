# Implementing MoE Architecture Operators on CANN Platform

## Overview

This repository implements **Mixture of Experts (MoE)** architecture operators optimized for the **Huawei Ascend CANN (Compute Architecture for Neural Networks)** platform. MoE is a powerful neural network architecture paradigm that enables scaling model capacity while keeping inference costs manageable through sparse activation of expert sub-networks.

## Key Features

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

## Project Structure

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

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Implementing-MoE-Architecture-Operators-on-CANN-Platform

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

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

## CANN Fused MoE

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

## Running Tests

```bash
pytest tests/ -v
```

## Running Examples

```bash
# Basic MoE demo
python src/examples/basic_moe.py

# Advanced MoE with CANN ops
python src/examples/advanced_moe.py
```

## References

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)
- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.