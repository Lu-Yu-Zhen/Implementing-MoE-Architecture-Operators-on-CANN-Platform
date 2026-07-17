"""Tests for MoE Expert modules."""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moe.experts import FeedForwardExpert, SharedExpert, ExpertGroup


class TestFeedForwardExpert:
    """Tests for FeedForwardExpert."""

    def test_forward_shape(self):
        expert = FeedForwardExpert(hidden_size=128, intermediate_size=512)
        x = torch.randn(2, 16, 128)
        out = expert(x)
        assert out.shape == x.shape

    def test_forward_2d_input(self):
        expert = FeedForwardExpert(hidden_size=128, intermediate_size=512)
        x = torch.randn(32, 128)
        out = expert(x)
        assert out.shape == x.shape

    def test_activations(self):
        for act in ["gelu", "relu", "silu"]:
            expert = FeedForwardExpert(hidden_size=64, activation=act)
            x = torch.randn(4, 8, 64)
            out = expert(x)
            assert out.shape == x.shape
            assert not torch.isnan(out).any()

    def test_dropout(self):
        expert = FeedForwardExpert(hidden_size=64, dropout=0.5)
        x = torch.randn(4, 8, 64)
        expert.train()
        out1 = expert(x)
        out2 = expert(x)
        # With dropout, outputs should differ in training
        assert not torch.allclose(out1, out2)


class TestSharedExpert:
    """Tests for SharedExpert."""

    def test_forward(self):
        expert = SharedExpert(hidden_size=128)
        x = torch.randn(2, 16, 128)
        out = expert(x)
        assert out.shape == x.shape


class TestExpertGroup:
    """Tests for ExpertGroup."""

    def test_forward_single(self):
        group = ExpertGroup(num_experts=4, hidden_size=128)
        x = torch.randn(8, 128)  # 8 tokens
        for i in range(4):
            out = group(x, i)
            assert out.shape == x.shape

    def test_forward_all(self):
        group = ExpertGroup(num_experts=4, hidden_size=128)
        x = torch.randn(8, 128)
        out = group.forward_all(x)
        assert out.shape == (4, 8, 128)

    def test_experts_independent(self):
        group = ExpertGroup(num_experts=4, hidden_size=64)
        x = torch.randn(8, 64)
        out0 = group(x, 0)
        out1 = group(x, 1)
        assert not torch.allclose(out0, out1)