"""
Reusable logging utility.

Configures the root logging handler exactly once (on first call), so
every module can call `get_logger()` and get a consistently formatted
logger without re-configuring `logging.basicConfig` themselves.
"""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_LOGGER_NAME = "hindu_research_agent"

_configured = False


def get_logger(name: str = _DEFAULT_LOGGER_NAME) -> logging.Logger:
    """Return the shared agent logger, configuring handlers on first use."""
    global _configured
    if not _configured:
        logging.basicConfig(
            level=logging.INFO,
            format=_LOG_FORMAT,
            datefmt=_DATE_FORMAT,
        )
        _configured = True
    return logging.getLogger(name)
