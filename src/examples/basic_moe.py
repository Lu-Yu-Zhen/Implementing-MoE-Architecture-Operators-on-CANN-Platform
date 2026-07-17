"""Basic MoE usage example on CANN platform.

Demonstrates:
1. Creating a basic MoE layer
2. Forward pass with routing
3. Load balancing loss computation
4. Device selection (NPU / CUDA / CPU)
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe import MoELayer, TopKRouter, FeedForwardExpert, LoadBalanceLoss
from moe.utils import get_device, is_npu_available


def main():
    # Detect device
    device = get_device()
    print(f"Using device: {device}")
    print(f"NPU available: {is_npu_available()}")

    # Configuration
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    num_experts = 8
    top_k = 2

    # Create MoE layer
    moe_layer = MoELayer(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        intermediate_size=4 * hidden_size,
        capacity_factor=1.25,
        use_shared_expert=False,
        jitter_noise=0.01,
    ).to(device)

    # Create load balance loss module
    lb_loss_fn = LoadBalanceLoss(num_experts=num_experts, loss_coefficient=0.01)

    # Dummy input
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    # Forward pass
    moe_layer.train()
    output, aux_loss = moe_layer(x)

    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary loss: {aux_loss.item():.6f}" if aux_loss is not None else "No aux loss")

    # Count parameters
    total_params = sum(p.numel() for p in moe_layer.parameters())
    trainable_params = sum(p.numel() for p in moe_layer.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Inference mode
    moe_layer.eval()
    with torch.no_grad():
        output_eval, _ = moe_layer(x)
    print(f"\nInference output shape: {output_eval.shape}")
    print(f"Output mean: {output_eval.mean().item():.6f}")
    print(f"Output std:  {output_eval.std().item():.6f}")


if __name__ == "__main__":
    main()