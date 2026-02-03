"""Performance metrics and timing utilities for InstantStyle."""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger("instant_style.metrics")

F = TypeVar("F", bound=Callable[..., Any])


def timed(func: F) -> F:
    """Decorator to log execution time of a function.

    Args:
        func: Function to wrap with timing

    Returns:
        Wrapped function that logs execution time
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start_time
            logger.info(f"{func.__qualname__} completed in {elapsed:.3f}s")

    return wrapper  # type: ignore[return-value]


class Timer:
    """Context manager for timing code blocks.

    Usage:
        with Timer("operation_name"):
            # code to time
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self.start_time
        logger.info(f"{self.name} completed in {self.elapsed:.3f}s")
