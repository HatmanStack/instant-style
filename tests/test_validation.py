"""Tests for input validation functions."""

import sys

import pytest
from PIL import Image

sys.path.insert(0, str(pytest.importorskip("pathlib").Path(__file__).parent.parent))

from exceptions import ValidationError
from validation import MAX_IMAGE_SIZE, MIN_IMAGE_SIZE, validate_inputs


class TestValidateInputs:
    """Test suite for validate_inputs function."""

    def test_valid_inputs(self, sample_image: Image.Image) -> None:
        """Test that valid inputs pass validation."""
        # Should not raise
        validate_inputs(
            image=sample_image,
            prompt="a beautiful landscape",
            scale=0.7,
            width=512,
            height=512,
        )

    def test_none_image_raises(self) -> None:
        """Test that None image raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                image=None,
                prompt="test prompt",
                scale=0.5,
                width=512,
                height=512,
            )
        assert exc_info.value.field == "image"
        assert "required" in str(exc_info.value).lower()

    def test_empty_prompt_raises(self, sample_image: Image.Image) -> None:
        """Test that empty prompt raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                image=sample_image,
                prompt="",
                scale=0.5,
                width=512,
                height=512,
            )
        assert exc_info.value.field == "prompt"

    def test_whitespace_prompt_raises(self, sample_image: Image.Image) -> None:
        """Test that whitespace-only prompt raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                image=sample_image,
                prompt="   \t\n  ",
                scale=0.5,
                width=512,
                height=512,
            )
        assert exc_info.value.field == "prompt"

    def test_scale_too_low_raises(self, sample_image: Image.Image) -> None:
        """Test that scale < 0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                image=sample_image,
                prompt="test prompt",
                scale=-0.1,
                width=512,
                height=512,
            )
        assert exc_info.value.field == "scale"

    def test_scale_too_high_raises(self, sample_image: Image.Image) -> None:
        """Test that scale > 1 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                image=sample_image,
                prompt="test prompt",
                scale=1.5,
                width=512,
                height=512,
            )
        assert exc_info.value.field == "scale"

    def test_scale_boundary_values(self, sample_image: Image.Image) -> None:
        """Test that scale boundary values (0 and 1) are valid."""
        # Scale = 0.0 should be valid
        validate_inputs(
            image=sample_image,
            prompt="test prompt",
            scale=0.0,
            width=512,
            height=512,
        )

        # Scale = 1.0 should be valid
        validate_inputs(
            image=sample_image,
            prompt="test prompt",
            scale=1.0,
            width=512,
            height=512,
        )

    def test_width_too_small_raises(self, sample_image: Image.Image) -> None:
        """Test that width below minimum raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                image=sample_image,
                prompt="test prompt",
                scale=0.5,
                width=MIN_IMAGE_SIZE - 1,
                height=512,
            )
        assert exc_info.value.field == "width"

    def test_width_too_large_raises(self, sample_image: Image.Image) -> None:
        """Test that width above maximum raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                image=sample_image,
                prompt="test prompt",
                scale=0.5,
                width=MAX_IMAGE_SIZE + 1,
                height=512,
            )
        assert exc_info.value.field == "width"

    def test_height_too_small_raises(self, sample_image: Image.Image) -> None:
        """Test that height below minimum raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                image=sample_image,
                prompt="test prompt",
                scale=0.5,
                width=512,
                height=MIN_IMAGE_SIZE - 1,
            )
        assert exc_info.value.field == "height"

    def test_height_too_large_raises(self, sample_image: Image.Image) -> None:
        """Test that height above maximum raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                image=sample_image,
                prompt="test prompt",
                scale=0.5,
                width=512,
                height=MAX_IMAGE_SIZE + 1,
            )
        assert exc_info.value.field == "height"

    def test_dimensions_not_divisible_by_8_raises(self, sample_image: Image.Image) -> None:
        """Test that dimensions not divisible by 8 raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                image=sample_image,
                prompt="test prompt",
                scale=0.5,
                width=513,  # Not divisible by 8
                height=512,
            )
        assert exc_info.value.field == "dimensions"

    def test_boundary_dimensions(self, sample_image: Image.Image) -> None:
        """Test boundary dimension values."""
        # Minimum valid dimensions
        validate_inputs(
            image=sample_image,
            prompt="test prompt",
            scale=0.5,
            width=MIN_IMAGE_SIZE,
            height=MIN_IMAGE_SIZE,
        )

        # Maximum valid dimensions
        validate_inputs(
            image=sample_image,
            prompt="test prompt",
            scale=0.5,
            width=MAX_IMAGE_SIZE,
            height=MAX_IMAGE_SIZE,
        )
