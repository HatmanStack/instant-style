"""Tests for custom exception hierarchy."""

import sys

import pytest

sys.path.insert(0, str(pytest.importorskip("pathlib").Path(__file__).parent.parent))

from exceptions import (
    ImageProcessingError,
    InferenceError,
    InstantStyleError,
    ModelLoadError,
    ResourceError,
    ValidationError,
)


class TestInstantStyleError:
    """Test suite for base InstantStyleError."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        error = InstantStyleError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.cause is None

    def test_error_with_cause(self) -> None:
        """Test error with underlying cause."""
        cause = ValueError("Original error")
        error = InstantStyleError("Wrapped error", cause=cause)
        assert str(error) == "Wrapped error"
        assert error.cause is cause

    def test_inheritance(self) -> None:
        """Test that InstantStyleError inherits from Exception."""
        error = InstantStyleError("test")
        assert isinstance(error, Exception)


class TestModelLoadError:
    """Test suite for ModelLoadError."""

    def test_basic_error(self) -> None:
        """Test basic ModelLoadError creation."""
        error = ModelLoadError("Failed to load model")
        assert str(error) == "Failed to load model"
        assert error.model_path is None
        assert error.cause is None

    def test_error_with_path(self) -> None:
        """Test ModelLoadError with model path."""
        error = ModelLoadError(
            "Failed to load model",
            model_path="/path/to/model.bin",
        )
        assert error.model_path == "/path/to/model.bin"

    def test_error_with_cause(self) -> None:
        """Test ModelLoadError with underlying cause."""
        cause = FileNotFoundError("File not found")
        error = ModelLoadError(
            "Failed to load model",
            model_path="/path/to/model.bin",
            cause=cause,
        )
        assert error.cause is cause
        assert error.model_path == "/path/to/model.bin"

    def test_inheritance(self) -> None:
        """Test that ModelLoadError inherits from InstantStyleError."""
        error = ModelLoadError("test")
        assert isinstance(error, InstantStyleError)
        assert isinstance(error, Exception)


class TestValidationError:
    """Test suite for ValidationError."""

    def test_basic_error(self) -> None:
        """Test basic ValidationError creation."""
        error = ValidationError("Invalid input")
        assert str(error) == "Invalid input"
        assert error.field is None

    def test_error_with_field(self) -> None:
        """Test ValidationError with field name."""
        error = ValidationError("Value out of range", field="scale")
        assert error.field == "scale"

    def test_inheritance(self) -> None:
        """Test that ValidationError inherits from InstantStyleError."""
        error = ValidationError("test")
        assert isinstance(error, InstantStyleError)


class TestImageProcessingError:
    """Test suite for ImageProcessingError."""

    def test_basic_error(self) -> None:
        """Test basic ImageProcessingError creation."""
        error = ImageProcessingError("Failed to process image")
        assert str(error) == "Failed to process image"

    def test_inheritance(self) -> None:
        """Test that ImageProcessingError inherits from InstantStyleError."""
        error = ImageProcessingError("test")
        assert isinstance(error, InstantStyleError)


class TestInferenceError:
    """Test suite for InferenceError."""

    def test_basic_error(self) -> None:
        """Test basic InferenceError creation."""
        error = InferenceError("Inference failed")
        assert str(error) == "Inference failed"

    def test_inheritance(self) -> None:
        """Test that InferenceError inherits from InstantStyleError."""
        error = InferenceError("test")
        assert isinstance(error, InstantStyleError)


class TestResourceError:
    """Test suite for ResourceError."""

    def test_basic_error(self) -> None:
        """Test basic ResourceError creation."""
        error = ResourceError("Out of memory")
        assert str(error) == "Out of memory"

    def test_inheritance(self) -> None:
        """Test that ResourceError inherits from InstantStyleError."""
        error = ResourceError("test")
        assert isinstance(error, InstantStyleError)


class TestExceptionCatching:
    """Test exception catching patterns."""

    def test_catch_all_with_base_class(self) -> None:
        """Test catching all custom exceptions with base class."""
        errors = [
            ModelLoadError("model"),
            ImageProcessingError("image"),
            ValidationError("validation"),
            InferenceError("inference"),
            ResourceError("resource"),
        ]

        for error in errors:
            try:
                raise error
            except InstantStyleError as e:
                assert isinstance(e, InstantStyleError)

    def test_catch_specific_exception(self) -> None:
        """Test catching specific exception type."""
        try:
            raise ValidationError("bad input", field="test")
        except ValidationError as e:
            assert e.field == "test"
        except InstantStyleError:
            pytest.fail("Should have caught ValidationError specifically")
