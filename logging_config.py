"""Structured logging configuration for InstantStyle."""

import logging
import sys


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> logging.Logger:
    """Configure structured logging for the application.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for logging output

    Returns:
        Configured logger instance
    """
    # Create formatter with timestamps
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    # Suppress noisy library loggers
    noisy_loggers = [
        "urllib3",
        "httpx",
        "httpcore",
        "diffusers",
        "transformers",
        "torch",
        "PIL",
        "filelock",
        "huggingface_hub",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Create and return application logger
    app_logger = logging.getLogger("instant_style")
    app_logger.setLevel(level)

    return app_logger


# Default logger instance
logger = setup_logging()
