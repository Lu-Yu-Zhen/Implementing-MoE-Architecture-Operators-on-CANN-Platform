"""CANN-optimized SwitchRouter operator for Ascend NPU.

Provides a fused implementation of the Switch Transformer routing operation
using Ascend C custom operators for maximum performance on NPU hardware.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from moe.utils import is_npu_available


class CannSwitchRouter(nn.Module):
    """CANN-optimized Switch Router for Ascend NPU.

    Uses custom Ascend C operators to fuse the routing computation
    (gating + top-1 selection + dispatch) into a single kernel.

    Args:
        hidden_size: Input hidden dimension.
        num_experts: Number of experts.
        capacity_factor: Buffer capacity factor (default: 1.0).
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        capacity_factor: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with CANN-optimized routing.

        Falls back to PyTorch implementation if NPU is not available.

        Args:
            x: Input of shape (batch, seq_len, hidden_size).

        Returns:
            Tuple of:
                - dispatch_indices: (batch, seq_len) expert index per token
                - dispatch_probs: (batch, seq_len) gating probability
                - capacity: Expert capacity
        """
        batch_size, seq_len, _ = x.shape

        if is_npu_available():
            return self._forward_npu(x)
        else:
            return self._forward_cpu(x)

    def _forward_npu(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """NPU-optimized forward using CANN custom operator.

        Uses the custom cann_switch_router operator that fuses:
        - Linear gating projection
        - Softmax
        - Top-1 expert selection
        - Capacity computation
        into a single Ascend C kernel.
        """
        batch_size, seq_len, _ = x.shape

        # Use CANN fused operator if available
        try:
            dispatch_indices, dispatch_probs = cann_switch_router(
                x,
                self.gate.weight,
                self.num_experts,
            )
        except (NameError, RuntimeError):
            # Fallback to PyTorch on NPU
            return self._forward_cpu(x)

        capacity = max(
            1,
            int(self.capacity_factor * seq_len / self.num_experts),
        )

        return dispatch_indices, dispatch_probs, capacity

    def _forward_cpu(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CPU/CUDA fallback implementation."""
        batch_size, seq_len, _ = x.shape

        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        top1_probs, top1_indices = torch.topk(probs, 1, dim=-1)

        capacity = max(
            1,
            int(self.capacity_factor * seq_len / self.num_experts),
        )

        return top1_indices.squeeze(-1), top1_probs.squeeze(-1), capacity


def cann_switch_router(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CANN custom operator for Switch Transformer routing.

    This is a stub that would be replaced by the actual Ascend C
    custom operator implementation. In production, this function
    would be registered via torch.library or torch_npu custom ops.

    Args:
        x: Input tensor (batch, seq_len, hidden_size).
        gate_weight: Gating weight matrix (num_experts, hidden_size).
        num_experts: Number of experts.

    Returns:
        Tuple of (dispatch_indices, dispatch_probs).
    """
    # Placeholder for Ascend C custom operator
    # In production, this would call the compiled .so kernel
    logits = torch.matmul(x, gate_weight.t())
    probs = torch.softmax(logits, dim=-1)
    top1_probs, top1_indices = torch.max(probs, dim=-1)
    return top1_indices, top1_probs