"""Device utility functions for MoE-CANN.

Handles device selection and NPU (Ascend) availability detection.
"""

import torch


def is_npu_available() -> bool:
    """Check if Ascend NPU is available via torch-npu.

    Returns:
        True if torch_npu is importable and NPU device is accessible.
    """
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except ImportError:
        return False


def get_device() -> torch.device:
    """Get the appropriate device (NPU > CUDA > CPU).

    Returns:
        torch.device for the best available hardware.
    """
    if is_npu_available():
        return torch.device("npu:0")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def set_device(device: str) -> None:
    """Set the default device for MoE operations.

    Args:
        device: Device string, one of 'npu', 'cuda', 'cpu'.
    """
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = device


_DEFAULT_DEVICE = "auto"


def _resolve_device() -> torch.device:
    """Resolve the default device based on configuration.

    Returns:
        Resolved torch.device.
    """
    if _DEFAULT_DEVICE == "auto":
        return get_device()
    return torch.device(_DEFAULT_DEVICE)