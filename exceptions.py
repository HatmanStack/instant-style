"""Custom exception hierarchy for InstantStyle."""


class InstantStyleError(Exception):
    """Base exception for all InstantStyle errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class ModelLoadError(InstantStyleError):
    """Raised when model loading fails."""

    def __init__(
        self, message: str, model_path: str | None = None, cause: Exception | None = None
    ) -> None:
        super().__init__(message, cause)
        self.model_path = model_path


class ImageProcessingError(InstantStyleError):
    """Raised when image processing fails."""

    pass


class ValidationError(InstantStyleError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class InferenceError(InstantStyleError):
    """Raised when model inference fails."""

    pass


class ResourceError(InstantStyleError):
    """Raised when resource allocation fails (e.g., GPU OOM)."""

    pass
