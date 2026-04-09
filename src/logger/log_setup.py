"""
src/logger/log_setup.py
────────────────────────
Centralised logging setup.
Call LoggerFactory.get_logger(__name__) in every module.
"""

import logging
import datetime
from pathlib import Path

import config


class LoggerFactory:
    """
    Factory class to create and configure named loggers.

    Usage:
        from src.logger.log_setup import LoggerFactory
        logger = LoggerFactory.get_logger(__name__)
    """

    _initialised: bool = False

    @classmethod
    def _initialise(cls) -> None:
        """Configure root logger once (idempotent)."""
        if cls._initialised:
            return

        log_file = config.LOG_DIR / f"rag_{datetime.date.today()}.log"

        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.LOG_LEVEL))

        formatter = logging.Formatter(
            fmt=config.LOG_FORMAT,
            datefmt=config.LOG_DATE_FORMAT,
        )

        # ── File handler ───────────────────────────────────────────────────────
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)

        # ── Console handler ────────────────────────────────────────────────────
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        cls._initialised = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Return a named logger. Triggers one-time root setup if needed.

        Args:
            name: Usually ``__name__`` of the calling module.

        Returns:
            logging.Logger: Configured logger instance.
        """
        cls._initialise()
        return logging.getLogger(name)
