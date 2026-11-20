"""
Live + historical financial data integration (Environment B data feed).

- Yahoo Finance via yfinance (delayed / end-of-day — never described as real-time)
- adjusted close handling (auto_adjust=True, as in the original notebook)
- in-memory TTL cache to avoid unnecessary API calls
- explicit source attribution, freshness timestamps and error handling
- never fabricates missing observations
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import DATA_SOURCE_LABEL

_CACHE: dict = {}
CACHE_TTL_SECONDS = 600          # 10-minute TTL for live quotes
HIST_CACHE_TTL_SECONDS = 3600    # 1-hour TTL for historical windows


class DataFetchError(RuntimeError):
    """Raised when a market-data source fails after retries."""


def _cache_get(key: str, ttl: int):
    item = _CACHE.get(key)
    if item and (time.time() - item['t']) < ttl:
        return item['data']
    return None


def _cache_put(key: str, data):
    _CACHE[key] = {'data': data, 't': time.time()}


def fetch_price_history(tickers: list[str], start: str, end: str,
                        force_refresh: bool = False) -> dict:
    """Download daily adjusted-close prices for a list of tickers.

    Returns {'prices': DataFrame, 'fetched_at': ISO timestamp, 'source': label}.
    Raises DataFetchError with a clean message on failure.
    """
    key = 'hist:' + hashlib.md5(
        f"{sorted(tickers)}|{start}|{end}".encode(),
        usedforsecurity=False).hexdigest()
    if not force_refresh:
        cached = _cache_get(key, HIST_CACHE_TTL_SECONDS)
        if cached is not None:
            return cached

    try:
        import yfinance as yf
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            raise DataFetchError(
                'No price data returned. Check your internet connection, '
                'ticker symbols, or try again later.')
        prices = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
        prices = prices.reindex(columns=tickers)
        prices = prices.ffill().dropna(how='all')
    except DataFetchError:
        raise
    except Exception as exc:                       # noqa: BLE001
        raise DataFetchError(
            f'Market-data download failed: {type(exc).__name__}. '
            'Yahoo Finance may be temporarily unavailable — please retry.') from None

    payload = {
        'prices': prices,
        'fetched_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'source': DATA_SOURCE_LABEL,
    }
    _cache_put(key, payload)
    return payload


def fetch_latest_quotes(tickers: list[str],
                        force_refresh: bool = False) -> dict:
    """Latest available end-of-day/delayed quote snapshot per ticker.

    Uses fast_info (last trade price, previous close) — clearly labelled as
    delayed data, never as tick-by-tick real-time.
    """
    key = 'quotes:' + hashlib.md5(str(sorted(tickers)).encode(),
                                   usedforsecurity=False).hexdigest()
    if not force_refresh:
        cached = _cache_get(key, CACHE_TTL_SECONDS)
        if cached is not None:
            return cached

    rows, errors = [], []
    try:
        import yfinance as yf
    except Exception:                              # noqa: BLE001
        raise DataFetchError('yfinance is not installed in this environment.') from None

    for t in tickers:
        try:
            fi = yf.Ticker(t).fast_info
            last = fi.get('last_price')
            prev = fi.get('previous_close')
            if last is None:
                errors.append(t)
                continue
            rows.append({
                'Ticker': t,
                'Last Price': float(last),
                'Prev Close': float(prev) if prev else np.nan,
                'Change %': (float(last) / float(prev) - 1.0) * 100
                            if prev else np.nan,
            })
        except Exception:                          # noqa: BLE001
            errors.append(t)

    if not rows:
        raise DataFetchError(
            'Live quotes are currently unavailable (data provider did not '
            'respond). Historical analysis is unaffected — please retry later.')

    payload = {
        'quotes': pd.DataFrame(rows),
        'failed_tickers': errors,
        'fetched_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'source': DATA_SOURCE_LABEL,
    }
    _cache_put(key, payload)
    return payload
