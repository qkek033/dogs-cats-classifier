"""
Logging configuration for the application.
"""
import logging
import logging.handlers
from pathlib import Path


def setup_logging(log_dir="logs"):
    """Configure root logger with file and console handlers."""
    Path(log_dir).mkdir(exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    inference_handler = logging.handlers.RotatingFileHandler(
        Path(log_dir) / "inference.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    inference_handler.setFormatter(formatter)
    root_logger.addHandler(inference_handler)

    error_handler = logging.handlers.RotatingFileHandler(
        Path(log_dir) / "errors.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    return root_logger
