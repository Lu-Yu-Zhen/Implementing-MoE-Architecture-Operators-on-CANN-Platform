"""Tests for MoE Router module."""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moe.router import TopKRouter, SwitchRouter, GatingNetwork


class TestGatingNetwork:
    """Tests for GatingNetwork."""

    def test_linear_gating(self):
        hidden_size, num_experts = 128, 8
        gate = GatingNetwork(hidden_size, num_experts)
        x = torch.randn(2, 16, hidden_size)
        logits = gate(x)
        assert logits.shape == (2, 16, num_experts)

    def test_mlp_gating(self):
        hidden_size, num_experts = 128, 8
        gate = GatingNetwork(hidden_size, num_experts, gating_hidden_size=64)
        x = torch.randn(2, 16, hidden_size)
        logits = gate(x)
        assert logits.shape == (2, 16, num_experts)


class TestTopKRouter:
    """Tests for TopKRouter."""

    def test_forward_shape(self):
        router = TopKRouter(hidden_size=128, num_experts=8, top_k=2)
        x = torch.randn(2, 16, 128)
        router.eval()
        dispatch_mask, combine_weights, expert_mask, aux_loss = router(x)

        assert dispatch_mask.dim() == 4  # (B, S, E, C)
        assert dispatch_mask.shape[0] == 2
        assert dispatch_mask.shape[1] == 16
        assert dispatch_mask.shape[2] == 8
        assert combine_weights.shape == dispatch_mask.shape
        assert expert_mask.shape == (2, 16, 8)

    def test_aux_loss_training(self):
        router = TopKRouter(hidden_size=128, num_experts=8, top_k=2)
        x = torch.randn(2, 16, 128)
        router.train()
        _, _, _, aux_loss = router(x)
        assert aux_loss is not None
        assert aux_loss.item() > 0

    def test_aux_loss_eval(self):
        router = TopKRouter(hidden_size=128, num_experts=8, top_k=2)
        x = torch.randn(2, 16, 128)
        router.eval()
        _, _, _, aux_loss = router(x)
        assert aux_loss is None

    def test_expert_selection(self):
        router = TopKRouter(hidden_size=128, num_experts=8, top_k=2)
        x = torch.randn(2, 16, 128)
        router.eval()
        _, _, expert_mask, _ = router(x)

        # Each token should be assigned to exactly top_k experts
        assigned = expert_mask.sum(dim=-1)
        assert (assigned <= 2).all()
        assert assigned.sum() > 0  # At least some tokens assigned


class TestSwitchRouter:
    """Tests for SwitchRouter."""

    def test_top1_routing(self):
        router = SwitchRouter(hidden_size=128, num_experts=8)
        x = torch.randn(2, 16, 128)
        router.eval()
        dispatch_mask, combine_weights, expert_mask, aux_loss = router(x)

        assert dispatch_mask.dim() == 4
        assert expert_mask.shape == (2, 16, 8)

        # Each token to at most 1 expert
        assigned = expert_mask.sum(dim=-1)
        assert (assigned <= 1).all()