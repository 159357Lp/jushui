"""Logging configuration."""

import logging
import sys
from pathlib import Path

from lmloader.config import settings


def setup_logging():
    log_config = settings.logging

    root_logger = logging.getLogger("lmloader")
    root_logger.setLevel(getattr(logging, log_config.level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_config.file:
        log_path = Path(log_config.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


logger = setup_logging()
