"""Load balancing utilities for MoE training.

Provides auxiliary loss functions and metrics to encourage balanced
expert utilization during training.
"""

import torch
import torch.nn as nn
from typing import Optional


def compute_load_balance_loss(
    gating_scores: torch.Tensor,
    expert_mask: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Compute the load balancing auxiliary loss.

    Measures the discrepancy between gating probabilities and actual
    expert assignment frequencies.

    Args:
        gating_scores: Softmax-normalized gating scores (batch, seq_len, num_experts).
        expert_mask: Binary mask of dispatched experts (batch, seq_len, num_experts).
        num_experts: Total number of experts.

    Returns:
        Scalar load balancing loss.
    """
    # Mean gating probability per expert
    mean_gate = gating_scores.mean(dim=[0, 1])  # (E,)

    # Fraction of tokens dispatched to each expert
    dispatched = expert_mask.float().sum(dim=[0, 1])  # (E,)
    total_tokens = dispatched.sum() + 1e-9
    fraction = dispatched / total_tokens  # (E,)

    loss = num_experts * (mean_gate * fraction).sum()
    return loss


def compute_expert_importance(
    gating_scores: torch.Tensor,
) -> torch.Tensor:
    """Compute importance score for each expert.

    Importance = sum of gating probabilities for each expert.

    Args:
        gating_scores: Softmax gating scores (batch, seq_len, num_experts).

    Returns:
        Importance vector of shape (num_experts,).
    """
    return gating_scores.sum(dim=[0, 1])


def compute_expert_utilization(
    expert_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute utilization rate for each expert.

    Utilization = fraction of tokens that use each expert.

    Args:
        expert_mask: Binary dispatch mask (batch, seq_len, num_experts).

    Returns:
        Utilization vector of shape (num_experts,).
    """
    total_tokens = expert_mask.shape[0] * expert_mask.shape[1]
    return expert_mask.float().sum(dim=[0, 1]) / total_tokens


class LoadBalanceLoss(nn.Module):
    """Load balancing loss module for MoE training.

    Wraps auxiliary loss computation in a convenient nn.Module interface.

    Args:
        num_experts: Number of experts.
        loss_coefficient: Weight for the auxiliary loss (default: 0.01).
    """

    def __init__(self, num_experts: int, loss_coefficient: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.loss_coefficient = loss_coefficient

    def forward(
        self,
        gating_scores: torch.Tensor,
        expert_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scaled load balancing loss.

        Args:
            gating_scores: Softmax gating scores.
            expert_mask: Binary dispatch mask.

        Returns:
            Scaled load balancing loss.
        """
        raw_loss = compute_load_balance_loss(
            gating_scores, expert_mask, self.num_experts
        )
        return self.loss_coefficient * raw_loss


def z_loss(logits: torch.Tensor) -> torch.Tensor:
    """Compute the z-loss (log-squared loss) for gating stability.

    Encourages gating logits to stay close to zero, preventing
    the router from producing extreme logit values.

    Args:
        logits: Raw gating logits from the router.

    Returns:
        Scalar z-loss.
    """
    return torch.mean(torch.square(torch.logsumexp(logits, dim=-1)))