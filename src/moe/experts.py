"""Expert network implementations for MoE architecture.

Provides standard FFN-based expert networks and shared expert variants.
"""

import torch
import torch.nn as nn
from typing import Optional


class FeedForwardExpert(nn.Module):
    """Standard feed-forward expert network used in MoE layers.

    Typically a 2-layer MLP with GELU activation, as used in
    most Transformer-based MoE architectures.

    Args:
        hidden_size: Input/output hidden dimension.
        intermediate_size: Intermediate FFN dimension (default: 4 * hidden_size).
        activation: Activation function ('gelu', 'relu', 'silu').
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        intermediate_size = intermediate_size or 4 * hidden_size

        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "silu":
            act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = act_fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert.

        Args:
            x: Input of shape (batch, seq_len, hidden_size) or (num_tokens, hidden_size).

        Returns:
            Output of same shape as input.
        """
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class SharedExpert(nn.Module):
    """Shared expert that all tokens pass through regardless of routing.

    Commonly used in DeepSeekMoE and similar architectures where a shared
    expert captures common knowledge across all tokens.

    Args:
        hidden_size: Input/output hidden dimension.
        intermediate_size: Intermediate FFN dimension.
        activation: Activation function.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.expert = FeedForwardExpert(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply shared expert to all tokens.

        Args:
            x: Input of shape (batch, seq_len, hidden_size).

        Returns:
            Output of same shape.
        """
        return self.expert(x)


class ExpertGroup(nn.Module):
    """Group of expert networks that can be dispatched to in parallel.

    Args:
        num_experts: Number of expert networks in the group.
        hidden_size: Hidden dimension for each expert.
        intermediate_size: Intermediate FFN dimension for each expert.
        activation: Activation function for all experts.
        dropout: Dropout rate for all experts.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size

        self.experts = nn.ModuleList([
            FeedForwardExpert(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation=activation,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Forward tokens through a specific expert.

        Args:
            x: Input tokens (num_tokens, hidden_size).
            expert_idx: Index of the expert to use.

        Returns:
            Expert output of shape (num_tokens, hidden_size).
        """
        return self.experts[expert_idx](x)

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through all experts and return stacked outputs.

        Args:
            x: Input of shape (num_tokens, hidden_size).

        Returns:
            Stacked expert outputs (num_experts, num_tokens, hidden_size).
        """
        outputs = []
        for expert in self.experts:
            outputs.append(expert(x))
        return torch.stack(outputs, dim=0)