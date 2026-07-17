"""MoE core module."""

from .router import TopKRouter, SwitchRouter, GatingNetwork
from .experts import FeedForwardExpert, SharedExpert, ExpertGroup
from .layers import MoELayer, SparseMoELayer
from .load_balance import LoadBalanceLoss, compute_load_balance_loss, compute_expert_utilization
from .utils import set_device, get_device, is_npu_available

__all__ = [
    "TopKRouter",
    "SwitchRouter",
    "GatingNetwork",
    "FeedForwardExpert",
    "SharedExpert",
    "ExpertGroup",
    "MoELayer",
    "SparseMoELayer",
    "LoadBalanceLoss",
    "compute_load_balance_loss",
    "compute_expert_utilization",
    "set_device",
    "get_device",
    "is_npu_available",
]