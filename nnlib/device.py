"""Device configuration helper.

Detects Apple Silicon GPU acceleration (MPS - Metal Performance Shaders)
with automatic fallback to standard CPU execution.
"""

from typing import Union
import torch


def get_device() -> torch.device:
    """Returns PyTorch device for Apple Silicon (MPS) if available, otherwise CPU.

    Returns:
        torch.device: 'mps' if Apple Silicon GPU is supported and available,
                      otherwise 'cpu'.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_name(device: Union[torch.device, str, None] = None) -> str:
    """Returns a friendly string describing the compute device.

    Args:
        device: The torch.device to describe. If None, queries get_device().

    Returns:
        str: Human-readable device description.
    """
    if device is None:
        device = get_device()

    dev_type = device.type if isinstance(device, torch.device) else str(device)
    if dev_type == "mps":
        return "Apple Silicon GPU (Metal Performance Shaders - MPS)"
    elif dev_type == "cuda":
        return f"NVIDIA GPU (CUDA) - {torch.cuda.get_device_name(0)}"
    return "Standard CPU"
