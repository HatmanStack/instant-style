"""Pytest fixtures for InstantStyle tests."""

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def device() -> str:
    """Get the available device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample RGB image for testing."""
    # Create a simple 256x256 gradient image
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            arr[i, j, 0] = i  # Red gradient
            arr[i, j, 1] = j  # Green gradient
            arr[i, j, 2] = 128  # Constant blue
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def small_image() -> Image.Image:
    """Create a small 64x64 image for testing."""
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def large_image() -> Image.Image:
    """Create a large 2048x2048 image for testing."""
    arr = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def mock_tensor() -> torch.Tensor:
    """Create a mock tensor for testing."""
    return torch.randn(1, 128, 3072)
