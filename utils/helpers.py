"""
Small, generic helper functions with no business logic of their own.
"""

from __future__ import annotations

from datetime import datetime, timezone


def current_utc_date_str(fmt: str = "%d %B %Y") -> str:
    """Return the current UTC date formatted with `fmt`."""
    return datetime.now(timezone.utc).strftime(fmt)
