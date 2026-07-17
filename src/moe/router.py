"""Router and Gating Network implementations for MoE.

Provides multiple routing strategies commonly used in MoE architectures:
- TopKRouter: Selects top-k experts based on gating scores.
- SwitchRouter: Switch Transformer style routing (top-1 per token).
- GatingNetwork: Configurable gating network (linear or MLP).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GatingNetwork(nn.Module):
    """Learned gating network that produces logits for expert selection.

    Supports both simple linear projection and deeper MLP gating.

    Args:
        hidden_size: Input hidden dimension.
        num_experts: Total number of experts.
        gating_hidden_size: Hidden size for MLP gating (None = linear gating).
        dropout: Dropout rate for gating MLP.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        gating_hidden_size: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        if gating_hidden_size is not None:
            self.gate = nn.Sequential(
                nn.Linear(hidden_size, gating_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(gating_hidden_size, num_experts, bias=False),
            )
        else:
            self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gating logits for input tokens.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Gating logits of shape (batch, seq_len, num_experts).
        """
        return self.gate(x)


class TopKRouter(nn.Module):
    """Top-K router that dispatches each token to the k experts with highest scores.

    Implements the standard noisy top-k gating from the paper
    "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer".

    Args:
        hidden_size: Input hidden dimension.
        num_experts: Total number of experts.
        top_k: Number of experts to route each token to.
        capacity_factor: Capacity factor for expert buffer (default: 1.25).
        gating_hidden_size: Hidden size for gating MLP (None = linear).
        jitter_noise: Standard deviation of gating noise during training.
        use_aux_loss: Whether to use auxiliary load balancing loss.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        gating_hidden_size: Optional[int] = None,
        jitter_noise: float = 0.0,
        use_aux_loss: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise
        self.use_aux_loss = use_aux_loss

        self.gating = GatingNetwork(
            hidden_size=hidden_size,
            num_experts=num_experts,
            gating_hidden_size=gating_hidden_size,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Route input tokens to top-k experts.

        Args:
            x: Input of shape (batch, seq_len, hidden_size).

        Returns:
            Tuple of:
                - dispatch_mask: (batch, seq_len, num_experts, capacity) float mask
                - combine_weights: (batch, seq_len, num_experts, capacity) float weights
                - expert_mask: (batch, seq_len, num_experts) bool mask
                - aux_loss: Scalar auxiliary load balancing loss (or None)
        """
        batch_size, seq_len, _ = x.shape

        # Compute gating logits
        logits = self.gating(x)

        # Add jitter noise during training
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(logits) * self.jitter_noise
            logits = logits + noise

        # Softmax over experts
        scores = F.softmax(logits, dim=-1)

        # Select top-k experts
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)

        # Normalize top-k scores
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute capacity per expert
        capacity = max(
            1,
            int(self.capacity_factor * seq_len * self.top_k / self.num_experts),
        )

        # Build dispatch mask and combine weights
        dispatch_mask = torch.zeros(
            batch_size, seq_len, self.num_experts, capacity,
            device=x.device, dtype=x.dtype,
        )
        combine_weights = torch.zeros_like(dispatch_mask)
        expert_mask = torch.zeros(
            batch_size, seq_len, self.num_experts,
            device=x.device, dtype=torch.bool,
        )

        # Expert capacity counters
        expert_counts = torch.zeros(
            batch_size, self.num_experts, device=x.device, dtype=torch.long,
        )

        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(self.top_k):
                    expert_idx = topk_indices[b, s, k].item()
                    count = expert_counts[b, expert_idx].item()
                    if count < capacity:
                        dispatch_mask[b, s, expert_idx, count] = 1.0
                        combine_weights[b, s, expert_idx, count] = topk_scores[b, s, k]
                        expert_mask[b, s, expert_idx] = True
                        expert_counts[b, expert_idx] += 1

        # Compute auxiliary load balancing loss
        aux_loss = None
        if self.use_aux_loss and self.training:
            aux_loss = self._compute_aux_loss(scores, expert_mask)

        return dispatch_mask, combine_weights, expert_mask, aux_loss

    def _compute_aux_loss(
        self, scores: torch.Tensor, expert_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary load balancing loss.

        Encourages uniform expert utilization.

        Args:
            scores: Gating scores of shape (batch, seq_len, num_experts).
            expert_mask: Binary mask of dispatched experts.

        Returns:
            Scalar auxiliary loss.
        """
        # Mean gating probability per expert
        mean_gate = scores.mean(dim=[0, 1])  # (num_experts,)

        # Fraction of tokens dispatched to each expert
        dispatched = expert_mask.float().sum(dim=[0, 1])  # (num_experts,)
        total_tokens = dispatched.sum() + 1e-9
        fraction = dispatched / total_tokens  # (num_experts,)

        # Aux loss = num_experts * sum(mean_gate_i * fraction_i)
        aux_loss = self.num_experts * (mean_gate * fraction).sum()
        return aux_loss


class SwitchRouter(nn.Module):
    """Switch Transformer style router (top-1 gating with a single expert per token).

    From "Switch Transformers: Scaling to Trillion Parameter Models with
    Simple and Efficient Sparsity".

    Args:
        hidden_size: Input hidden dimension.
        num_experts: Total number of experts.
        capacity_factor: Expert capacity factor (default: 1.0).
        gating_hidden_size: Optional hidden size for gating MLP.
        jitter_noise: Gating noise standard deviation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        gating_hidden_size: Optional[int] = None,
        jitter_noise: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise

        self.gating = GatingNetwork(
            hidden_size=hidden_size,
            num_experts=num_experts,
            gating_hidden_size=gating_hidden_size,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Route each token to the single best expert (Switch routing).

        Args:
            x: Input of shape (batch, seq_len, hidden_size).

        Returns:
            Same tuple format as TopKRouter.
        """
        batch_size, seq_len, _ = x.shape

        logits = self.gating(x)

        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(logits) * self.jitter_noise
            logits = logits + noise

        scores = F.softmax(logits, dim=-1)

        # Select top-1 expert
        top1_scores, top1_indices = torch.topk(scores, 1, dim=-1)
        top1_scores = top1_scores.squeeze(-1)  # (batch, seq_len)
        top1_indices = top1_indices.squeeze(-1)  # (batch, seq_len)

        capacity = max(
            1,
            int(self.capacity_factor * seq_len / self.num_experts),
        )

        dispatch_mask = torch.zeros(
            batch_size, seq_len, self.num_experts, capacity,
            device=x.device, dtype=x.dtype,
        )
        combine_weights = torch.zeros_like(dispatch_mask)
        expert_mask = torch.zeros(
            batch_size, seq_len, self.num_experts,
            device=x.device, dtype=torch.bool,
        )

        expert_counts = torch.zeros(
            batch_size, self.num_experts, device=x.device, dtype=torch.long,
        )

        for b in range(batch_size):
            for s in range(seq_len):
                expert_idx = top1_indices[b, s].item()
                count = expert_counts[b, expert_idx].item()
                if count < capacity:
                    dispatch_mask[b, s, expert_idx, count] = 1.0
                    combine_weights[b, s, expert_idx, count] = 1.0  # Switch uses weight=1
                    expert_mask[b, s, expert_idx] = True
                    expert_counts[b, expert_idx] += 1

        # Aux loss for load balancing
        aux_loss = None
        if self.training:
            mean_gate = scores.mean(dim=[0, 1])
            dispatched = expert_mask.float().sum(dim=[0, 1])
            total_tokens = dispatched.sum() + 1e-9
            fraction = dispatched / total_tokens
            aux_loss = self.num_experts * (mean_gate * fraction).sum()

        return dispatch_mask, combine_weights, expert_mask, aux_loss