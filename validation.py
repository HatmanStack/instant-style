"""Input validation utilities for InstantStyle."""

from PIL import Image

from exceptions import ValidationError

# Constants
MAX_IMAGE_SIZE = 1024
MIN_IMAGE_SIZE = 256


def validate_inputs(
    image: Image.Image | None,
    prompt: str,
    scale: float,
    width: int,
    height: int,
) -> None:
    """Validate user inputs before processing.

    Args:
        image: Input image (required)
        prompt: Text prompt
        scale: IP-Adapter scale factor
        width: Output width
        height: Output height

    Raises:
        ValidationError: If any input is invalid
    """
    if image is None:
        raise ValidationError("Image is required", field="image")

    if not prompt or not prompt.strip():
        raise ValidationError("Prompt cannot be empty", field="prompt")

    if not 0.0 <= scale <= 1.0:
        raise ValidationError(f"Scale must be between 0.0 and 1.0, got {scale}", field="scale")

    if not MIN_IMAGE_SIZE <= width <= MAX_IMAGE_SIZE:
        raise ValidationError(
            f"Width must be between {MIN_IMAGE_SIZE} and {MAX_IMAGE_SIZE}, got {width}",
            field="width",
        )

    if not MIN_IMAGE_SIZE <= height <= MAX_IMAGE_SIZE:
        raise ValidationError(
            f"Height must be between {MIN_IMAGE_SIZE} and {MAX_IMAGE_SIZE}, got {height}",
            field="height",
        )

    if width % 8 != 0 or height % 8 != 0:
        raise ValidationError(
            "Width and height must be divisible by 8",
            field="dimensions",
        )
