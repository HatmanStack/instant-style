"""Tests for attention processor module."""

import sys

import pytest
import torch

sys.path.insert(0, str(pytest.importorskip("pathlib").Path(__file__).parent.parent))

from attention_processor import IPAFluxAttnProcessor2_0


class TestIPAFluxAttnProcessor:
    """Test suite for IPAFluxAttnProcessor2_0."""

    @pytest.fixture
    def processor(self, device: str) -> IPAFluxAttnProcessor2_0:
        """Create a test processor."""
        return IPAFluxAttnProcessor2_0(
            hidden_size=3072,
            cross_attention_dim=4096,
            scale=1.0,
            num_tokens=128,
        ).to(device)

    def test_init_creates_layers(self, processor: IPAFluxAttnProcessor2_0) -> None:
        """Test that initialization creates the expected layers."""
        assert processor.hidden_size == 3072
        assert processor.cross_attention_dim == 4096
        assert processor.scale == 1.0
        assert processor.num_tokens == 128
        assert hasattr(processor, "to_k_ip")
        assert hasattr(processor, "to_v_ip")
        assert hasattr(processor, "norm_added_k")

    def test_default_scale(self, device: str) -> None:
        """Test that default scale is 1.0."""
        processor = IPAFluxAttnProcessor2_0(
            hidden_size=3072,
            cross_attention_dim=4096,
            num_tokens=4,
        ).to(device)
        assert processor.scale == 1.0

    def test_linear_layer_shapes(self, processor: IPAFluxAttnProcessor2_0) -> None:
        """Test that linear layers have correct input/output dimensions."""
        # to_k_ip should project from cross_attention_dim to hidden_size
        assert processor.to_k_ip.in_features == 4096
        assert processor.to_k_ip.out_features == 3072

        # to_v_ip should project from cross_attention_dim to hidden_size
        assert processor.to_v_ip.in_features == 4096
        assert processor.to_v_ip.out_features == 3072

    def test_k_projection(self, processor: IPAFluxAttnProcessor2_0, device: str) -> None:
        """Test key projection forward pass."""
        batch_size = 2
        seq_len = 128
        input_tensor = torch.randn(batch_size, seq_len, 4096, device=device)

        output = processor.to_k_ip(input_tensor)

        assert output.shape == (batch_size, seq_len, 3072)

    def test_v_projection(self, processor: IPAFluxAttnProcessor2_0, device: str) -> None:
        """Test value projection forward pass."""
        batch_size = 2
        seq_len = 128
        input_tensor = torch.randn(batch_size, seq_len, 4096, device=device)

        output = processor.to_v_ip(input_tensor)

        assert output.shape == (batch_size, seq_len, 3072)

    def test_norm_layer(self, processor: IPAFluxAttnProcessor2_0, device: str) -> None:
        """Test normalization layer."""
        batch_size = 2
        num_heads = 24
        seq_len = 128
        head_dim = 128

        # Simulate projected key tensor after reshaping
        input_tensor = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        output = processor.norm_added_k(input_tensor)

        assert output.shape == input_tensor.shape

    def test_different_scales(self, device: str) -> None:
        """Test creating processors with different scales."""
        for scale in [0.0, 0.5, 1.0]:
            processor = IPAFluxAttnProcessor2_0(
                hidden_size=3072,
                cross_attention_dim=4096,
                scale=scale,
                num_tokens=4,
            ).to(device)
            assert processor.scale == scale

    def test_different_num_tokens(self, device: str) -> None:
        """Test creating processors with different num_tokens."""
        for num_tokens in [4, 16, 64, 128]:
            processor = IPAFluxAttnProcessor2_0(
                hidden_size=3072,
                cross_attention_dim=4096,
                scale=1.0,
                num_tokens=num_tokens,
            ).to(device)
            assert processor.num_tokens == num_tokens


class TestIPAFluxAttnProcessorDtypes:
    """Test IPAFluxAttnProcessor2_0 with different data types."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dtype_preservation(self, dtype: torch.dtype, device: str) -> None:
        """Test that processor preserves input dtype."""
        if dtype == torch.bfloat16 and device == "cpu":
            pytest.skip("bfloat16 may not be fully supported on CPU")

        processor = IPAFluxAttnProcessor2_0(
            hidden_size=3072,
            cross_attention_dim=4096,
            scale=1.0,
            num_tokens=4,
        ).to(device, dtype=dtype)

        input_tensor = torch.randn(1, 4, 4096, device=device, dtype=dtype)

        k_output = processor.to_k_ip(input_tensor)
        v_output = processor.to_v_ip(input_tensor)

        assert k_output.dtype == dtype
        assert v_output.dtype == dtype
