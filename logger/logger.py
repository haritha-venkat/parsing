import logging
import datetime
from pathlib import Path

from config import LOG_DIR, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT

_LOG_FILE: Path = LOG_DIR / f"rag_{datetime.date.today()}.log"
_CONFIGURED: bool = False


def setup_logging() -> None:
    """
    Configure the root logger once.
    Subsequent calls are no-ops (guarded by _CONFIGURED flag).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _CONFIGURED = True
    root_logger.info("Logging initialised → %s", _LOG_FILE)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger, ensuring logging is set up first.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        logging.Logger: Configured logger instance.
    """
    setup_logging()
    return logging.getLogger(name)
