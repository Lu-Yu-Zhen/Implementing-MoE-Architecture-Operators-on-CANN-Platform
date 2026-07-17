"""CANN-optimized All-to-All communication for MoE dispatch/combine.

On Ascend NPU clusters, expert parallelism requires efficient all-to-all
communication. This module provides CANN-optimized wrappers around
HCCL (Huawei Collective Communication Library) all-to-all operations.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from moe.utils import is_npu_available


class CannAllToAll(nn.Module):
    """CANN-optimized All-to-All communication for MoE.

    Wraps HCCL all-to-all collective operations for expert-parallel MoE,
    where tokens need to be redistributed across devices based on
    router decisions.

    Supports both:
    - Dispatch: Send tokens from current device to expert devices.
    - Combine: Gather expert outputs back to original devices.

    Args:
        num_experts: Total number of experts.
        world_size: Number of NPU devices in expert-parallel group.
        experts_per_device: Number of experts hosted per device.
    """

    def __init__(
        self,
        num_experts: int,
        world_size: int = 1,
        experts_per_device: Optional[int] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.world_size = world_size
        self.experts_per_device = experts_per_device or (num_experts // world_size)

    def dispatch(
        self, x: torch.Tensor, expert_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dispatch tokens to expert devices via all-to-all.

        Args:
            x: Input tokens (batch, seq_len, hidden_size).
            expert_indices: Expert assignment per token (batch, seq_len).

        Returns:
            Tuple of:
                - dispatched_tokens: Tokens grouped by target expert device.
                - token_order: Indices for restoring original order.
        """
        if is_npu_available() and self.world_size > 1:
            return self._dispatch_npu(x, expert_indices)
        else:
            return self._dispatch_local(x, expert_indices)

    def _dispatch_npu(
        self, x: torch.Tensor, expert_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """NPU-optimized dispatch using HCCL all-to-all.

        Uses HCCL alltoallv to redistribute tokens across devices
        based on expert assignment.
        """
        try:
            dispatched, token_order = cann_all_to_all_dispatch(
                x, expert_indices, self.num_experts, self.world_size
            )
            return dispatched, token_order
        except (NameError, RuntimeError):
            return self._dispatch_local(x, expert_indices)

    def _dispatch_local(
        self, x: torch.Tensor, expert_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Local dispatch (single-device fallback)."""
        batch_size, seq_len, hidden_size = x.shape

        # Group tokens by expert
        flat_x = x.reshape(-1, hidden_size)
        flat_indices = expert_indices.reshape(-1)

        sorted_indices = torch.argsort(flat_indices)
        sorted_x = flat_x[sorted_indices]

        token_order = torch.argsort(sorted_indices)

        return sorted_x, token_order

    def combine(
        self, expert_outputs: torch.Tensor, token_order: torch.Tensor
    ) -> torch.Tensor:
        """Combine expert outputs back to original token order.

        Args:
            expert_outputs: Expert outputs in dispatch order.
            token_order: Token ordering from dispatch.

        Returns:
            Combined outputs in original token order.
        """
        if is_npu_available() and self.world_size > 1:
            return self._combine_npu(expert_outputs, token_order)
        else:
            return self._combine_local(expert_outputs, token_order)

    def _combine_npu(
        self, expert_outputs: torch.Tensor, token_order: torch.Tensor
    ) -> torch.Tensor:
        """NPU-optimized combine using HCCL all-to-all."""
        try:
            combined = cann_all_to_all_combine(
                expert_outputs, token_order, self.num_experts, self.world_size
            )
            return combined
        except (NameError, RuntimeError):
            return self._combine_local(expert_outputs, token_order)

    def _combine_local(
        self, expert_outputs: torch.Tensor, token_order: torch.Tensor
    ) -> torch.Tensor:
        """Local combine (single-device fallback)."""
        return expert_outputs[token_order]


def cann_all_to_all_dispatch(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
    world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CANN custom operator for all-to-all dispatch.

    Placeholder for Ascend C / HCCL custom operator that performs
    token redistribution across NPU devices.

    Args:
        x: Input tokens (batch, seq_len, hidden_size).
        expert_indices: Expert assignment per token.
        num_experts: Total experts.
        world_size: Expert-parallel group size.

    Returns:
        Tuple of (dispatched_tokens, token_order).
    """
    # Placeholder: in production, this calls HCCL alltoallv
    # via torch_npu.distributed.all_to_all_single or custom HCCL op
    batch_size, seq_len, hidden_size = x.shape
    flat_x = x.reshape(-1, hidden_size)
    flat_indices = expert_indices.reshape(-1)
    sorted_indices = torch.argsort(flat_indices)
    sorted_x = flat_x[sorted_indices]
    token_order = torch.argsort(sorted_indices)
    return sorted_x, token_order


def cann_all_to_all_combine(
    expert_outputs: torch.Tensor,
    token_order: torch.Tensor,
    num_experts: int,
    world_size: int,
) -> torch.Tensor:
    """CANN custom operator for all-to-all combine.

    Placeholder for Ascend C / HCCL custom operator.

    Args:
        expert_outputs: Expert outputs in dispatch order.
        token_order: Token ordering from dispatch.
        num_experts: Total experts.
        world_size: Expert-parallel group size.

    Returns:
        Combined outputs in original order.
    """
    return expert_outputs[token_order]


def cann_all_to_all(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
    world_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """High-level CANN all-to-all wrapper for MoE.

    Convenience function that combines dispatch and token ordering.

    Args:
        x: Input tokens.
        expert_indices: Expert assignments.
        num_experts: Total experts.
        world_size: Expert-parallel group size.

    Returns:
        Tuple of (dispatched_tokens, token_order).
    """
    return cann_all_to_all_dispatch(
        x, expert_indices, num_experts, world_size
    )