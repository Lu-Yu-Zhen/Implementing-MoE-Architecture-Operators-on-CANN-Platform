"""Advanced MoE example with CANN fused operators.

Demonstrates:
1. Switch Transformer style routing
2. CANN fused MoE layer
3. Shared expert configuration
4. Multi-expert load balancing analysis
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe import (
    SparseMoELayer,
    MoELayer,
    TopKRouter,
    SwitchRouter,
    FeedForwardExpert,
    SharedExpert,
    ExpertGroup,
    compute_load_balance_loss,
    compute_expert_utilization,
)
from moe.utils import get_device, is_npu_available


def demo_switch_transformer():
    """Demonstrate Switch Transformer style routing (top-1 per token)."""
    print("=" * 60)
    print("Switch Transformer MoE Demo")
    print("=" * 60)

    device = get_device()
    batch_size, seq_len, hidden_size = 4, 32, 512
    num_experts = 8

    layer = SparseMoELayer(
        hidden_size=hidden_size,
        num_experts=num_experts,
        intermediate_size=4 * hidden_size,
        capacity_factor=1.0,
        jitter_noise=0.01,
    ).to(device)

    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    layer.train()
    output, aux_loss = layer(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Aux loss: {aux_loss.item():.6f}" if aux_loss is not None else "No aux loss")

    params = sum(p.numel() for p in layer.parameters())
    print(f"Parameters: {params:,}")


def demo_shared_expert():
    """Demonstrate MoE with a shared expert (DeepSeekMoE style)."""
    print("\n" + "=" * 60)
    print("MoE with Shared Expert Demo")
    print("=" * 60)

    device = get_device()
    batch_size, seq_len, hidden_size = 2, 16, 256
    num_experts = 6

    layer = MoELayer(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=2,
        intermediate_size=4 * hidden_size,
        use_shared_expert=True,
        dropout=0.1,
    ).to(device)

    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    layer.train()
    output, aux_loss = layer(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Shared expert enabled: {layer.shared_expert is not None}")
    print(f"Aux loss: {aux_loss.item():.6f}" if aux_loss is not None else "No aux loss")


def demo_load_balance_analysis():
    """Analyze expert utilization and load balance."""
    print("\n" + "=" * 60)
    print("Load Balance Analysis Demo")
    print("=" * 60)

    device = get_device()
    batch_size, seq_len, hidden_size = 4, 64, 256
    num_experts = 8

    router = TopKRouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=2,
        jitter_noise=0.1,
    ).to(device)

    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    router.train()

    dispatch_mask, combine_weights, expert_mask, aux_loss = router(x)

    # Compute gating scores
    logits = router.gating(x)
    scores = torch.softmax(logits, dim=-1)

    utilization = compute_expert_utilization(expert_mask)
    lb_loss = compute_load_balance_loss(scores, expert_mask, num_experts)

    print("Expert utilization:")
    for i, util in enumerate(utilization.tolist()):
        bar = "#" * int(util * 50)
        print(f"  Expert {i}: {util:.4f} |{bar}")

    print(f"\nLoad balance loss: {lb_loss.item():.6f}")
    print(f"Mean utilization: {utilization.mean().item():.4f}")
    print(f"Std utilization:  {utilization.std().item():.4f}")


def demo_cann_fused():
    """Demonstrate CANN fused MoE (falls back to PyTorch if NPU unavailable)."""
    print("\n" + "=" * 60)
    print("CANN Fused MoE Demo")
    print("=" * 60)

    # Import CANN ops
    from cann_ops.fused_moe import CannFusedMoE

    device = get_device()
    print(f"Device: {device} (NPU: {is_npu_available()})")

    batch_size, seq_len, hidden_size = 2, 16, 256
    num_experts = 8

    layer = CannFusedMoE(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=2,
        capacity_factor=1.25,
    ).to(device)

    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    if is_npu_available():
        print("Attempting CANN fused kernel...")
    else:
        print("NPU not available, using PyTorch fallback...")

    layer.train()
    output, aux_loss = layer(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Aux loss: {aux_loss.item():.6f}" if aux_loss is not None else "No aux loss")

    params = sum(p.numel() for p in layer.parameters())
    print(f"Parameters: {params:,}")


if __name__ == "__main__":
    demo_switch_transformer()
    demo_shared_expert()
    demo_load_balance_analysis()
    demo_cann_fused()
    print("\n" + "=" * 60)
    print("All demos completed successfully!")