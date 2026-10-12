"""Tests for hardware device detection (Apple Silicon MPS / CPU fallback)."""

import torch
from nnlib.device import get_device, get_device_name


def test_get_device_returns_valid_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ["mps", "cpu", "cuda"]


def test_device_name_string():
    name = get_device_name()
    assert isinstance(name, str)
    assert len(name) > 0


def test_tensor_allocation_on_device():
    device = get_device()
    t = torch.ones((3, 3), device=device)
    assert t.device.type == device.type
    # Simple operation on device
    t_double = t * 2.0
    assert torch.allclose(t_double.cpu(), torch.full((3, 3), 2.0))
