"""Tests for MoE Layer module."""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moe.layers import MoELayer, SparseMoELayer
from moe.utils import get_device


class TestMoELayer:
    """Tests for MoELayer."""

    def test_forward_shape(self):
        layer = MoELayer(hidden_size=128, num_experts=4, top_k=2)
        layer.eval()
        x = torch.randn(2, 16, 128)
        output, aux_loss = layer(x)
        assert output.shape == x.shape
        assert aux_loss is None  # eval mode

    def test_forward_training(self):
        layer = MoELayer(hidden_size=128, num_experts=4, top_k=2)
        layer.train()
        x = torch.randn(2, 16, 128)
        output, aux_loss = layer(x)
        assert output.shape == x.shape
        assert aux_loss is not None

    def test_with_shared_expert(self):
        layer = MoELayer(
            hidden_size=128, num_experts=4, top_k=2, use_shared_expert=True
        )
        layer.eval()
        x = torch.randn(2, 16, 128)
        output, _ = layer(x)
        assert output.shape == x.shape

    def test_gradient_flow(self):
        layer = MoELayer(hidden_size=64, num_experts=4, top_k=2)
        layer.train()
        x = torch.randn(2, 8, 64)
        output, aux_loss = layer(x)

        loss = output.sum()
        if aux_loss is not None:
            loss = loss + aux_loss
        loss.backward()

        # Check gradients exist
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestSparseMoELayer:
    """Tests for SparseMoELayer."""

    def test_forward_shape(self):
        layer = SparseMoELayer(hidden_size=128, num_experts=4)
        layer.eval()
        x = torch.randn(2, 16, 128)
        output, aux_loss = layer(x)
        assert output.shape == x.shape

    def test_forward_training(self):
        layer = SparseMoELayer(hidden_size=128, num_experts=4)
        layer.train()
        x = torch.randn(2, 16, 128)
        output, aux_loss = layer(x)
        assert output.shape == x.shape
        assert aux_loss is not None

    def test_device(self):
        """Test that layer works on the detected device."""
        device = get_device()
        layer = SparseMoELayer(hidden_size=64, num_experts=4).to(device)
        layer.eval()
        x = torch.randn(2, 8, 64, device=device)
        output, _ = layer(x)
        assert output.device == device