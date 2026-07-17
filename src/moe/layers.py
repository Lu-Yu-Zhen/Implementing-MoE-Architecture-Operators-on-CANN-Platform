"""MoE Layer implementations combining routing and expert computation.

Provides:
- MoELayer: Full MoE layer with routing, expert dispatch, and output combination.
- SparseMoELayer: Memory-efficient sparse MoE layer.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .router import TopKRouter, SwitchRouter
from .experts import ExpertGroup, SharedExpert


class MoELayer(nn.Module):
    """Full Mixture of Experts layer.

    Replaces a standard FFN with multiple expert FFNs and a learned router.

    Args:
        hidden_size: Model hidden dimension.
        num_experts: Number of expert networks.
        top_k: Number of experts per token.
        intermediate_size: Expert FFN intermediate size (default: 4 * hidden_size).
        capacity_factor: Buffer capacity factor for expert dispatch.
        gating_hidden_size: Optional hidden size for gating MLP.
        use_shared_expert: Whether to include a shared expert.
        jitter_noise: Gating noise during training.
        dropout: Dropout rate for experts.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        intermediate_size: Optional[int] = None,
        capacity_factor: float = 1.25,
        gating_hidden_size: Optional[int] = None,
        use_shared_expert: bool = False,
        jitter_noise: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_shared_expert = use_shared_expert

        # Router
        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            gating_hidden_size=gating_hidden_size,
            jitter_noise=jitter_noise,
        )

        # Expert group
        self.experts = ExpertGroup(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )

        # Optional shared expert
        if use_shared_expert:
            self.shared_expert = SharedExpert(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dropout=dropout,
            )
        else:
            self.shared_expert = None

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through MoE layer.

        Args:
            x: Input of shape (batch, seq_len, hidden_size).

        Returns:
            Tuple of:
                - output: (batch, seq_len, hidden_size)
                - aux_loss: Scalar auxiliary load balancing loss (or None)
        """
        batch_size, seq_len, hidden_size = x.shape

        # Route tokens
        dispatch_mask, combine_weights, expert_mask, aux_loss = self.router(x)

        # Combined output
        output = torch.zeros_like(x)

        # Dispatch tokens to experts and combine
        capacity = dispatch_mask.shape[-1]

        for expert_idx in range(self.num_experts):
            # Find tokens dispatched to this expert
            expert_dispatch = dispatch_mask[:, :, expert_idx, :]  # (B, S, C)
            expert_weights = combine_weights[:, :, expert_idx, :]  # (B, S, C)

            # Collect tokens
            token_mask = expert_dispatch.sum(dim=-1) > 0  # (B, S)
            if token_mask.sum() == 0:
                continue

            # Gather tokens
            b_indices, s_indices = torch.where(token_mask)
            tokens = x[b_indices, s_indices]  # (num_tokens, H)

            # Forward through expert
            expert_out = self.experts(tokens, expert_idx)  # (num_tokens, H)

            # Scatter back with combine weights
            for i, (b, s) in enumerate(zip(b_indices.tolist(), s_indices.tolist())):
                # Find the capacity slot for this token
                cap_slot = torch.where(expert_dispatch[b, s] > 0)[0]
                for cs in cap_slot:
                    w = expert_weights[b, s, cs.item()]
                    output[b, s] += w * expert_out[i]

        # Add shared expert if enabled
        if self.shared_expert is not None:
            output = output + self.shared_expert(x)

        return output, aux_loss


class SparseMoELayer(nn.Module):
    """Sparse MoE Layer using Switch Transformer routing (top-1 per token).

    More memory-efficient than full MoE since each token only goes to one expert.

    Args:
        hidden_size: Model hidden dimension.
        num_experts: Number of expert networks.
        intermediate_size: Expert FFN intermediate size.
        capacity_factor: Buffer capacity factor.
        gating_hidden_size: Optional hidden size for gating MLP.
        jitter_noise: Gating noise during training.
        dropout: Dropout rate for experts.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 8,
        intermediate_size: Optional[int] = None,
        capacity_factor: float = 1.0,
        gating_hidden_size: Optional[int] = None,
        jitter_noise: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        self.router = SwitchRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            gating_hidden_size=gating_hidden_size,
            jitter_noise=jitter_noise,
        )

        self.experts = ExpertGroup(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through sparse MoE layer.

        Args:
            x: Input of shape (batch, seq_len, hidden_size).

        Returns:
            Tuple of output and auxiliary loss.
        """
        batch_size, seq_len, hidden_size = x.shape

        dispatch_mask, combine_weights, expert_mask, aux_loss = self.router(x)
        capacity = dispatch_mask.shape[-1]

        output = torch.zeros_like(x)

        for expert_idx in range(self.num_experts):
            expert_dispatch = dispatch_mask[:, :, expert_idx, :]
            token_mask = expert_dispatch.sum(dim=-1) > 0

            if token_mask.sum() == 0:
                continue

            b_indices, s_indices = torch.where(token_mask)
            tokens = x[b_indices, s_indices]
            expert_out = self.experts(tokens, expert_idx)

            for i, (b, s) in enumerate(zip(b_indices.tolist(), s_indices.tolist())):
                output[b, s] = expert_out[i]  # Switch uses weight=1

        return output, aux_loss