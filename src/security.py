"""Security controls module — input validation and safe output handling.

Implements the security checklist from the GWP3 deployment specification:
- strict input validation for tickers, dates and API-ish parameters
- no eval/exec/shell, no pickle deserialization, no user-uploaded code paths
- errors are logged generically; tracebacks and internal paths are never shown
"""
from __future__ import annotations

import re
from datetime import date

from .config import TEST_END, TEST_START, TRAIN_END, TRAIN_START

# Strict ticker allow-list: uppercase letters, digits, dot or dash, 1-10 chars.
_TICKER_RE = re.compile(r'^[A-Z0-9.\-]{1,10}$')

# Known-good date range boundaries for this application.
_MIN_DATE = date(2020, 1, 1)
_MAX_DATE = date(2035, 12, 31)


class InputValidationError(ValueError):
    """Raised when user input fails validation (safe, generic message)."""


def validate_tickers(tickers: list[str]) -> list[str]:
    """Return the subset of tickers that pass the strict pattern."""
    if not isinstance(tickers, list) or not tickers or len(tickers) > 50:
        raise InputValidationError('Invalid ticker selection.')
    cleaned = []
    for t in tickers:
        if not isinstance(t, str) or not _TICKER_RE.match(t):
            raise InputValidationError('Invalid ticker symbol.')
        cleaned.append(t)
    return cleaned


def validate_date_range(start: str, end: str) -> tuple[str, str]:
    """Validate ISO date strings and their ordering against sane bounds."""
    try:
        s = date.fromisoformat(str(start))
        e = date.fromisoformat(str(end))
    except (TypeError, ValueError) as exc:
        raise InputValidationError('Invalid date format.') from exc
    if not (_MIN_DATE <= s <= _MAX_DATE and _MIN_DATE <= e <= _MAX_DATE):
        raise InputValidationError('Dates out of allowed range.')
    if s >= e:
        raise InputValidationError('Start date must precede end date.')
    return str(s), str(e)


def safe_error_message(exc: Exception) -> str:
    """Map an internal exception to a user-safe message with no internals."""
    known = (InputValidationError, ConnectionError, TimeoutError)
    if isinstance(exc, known):
        return str(exc)
    return ('An unexpected error occurred while processing your request. '
            'Please try again later.')
