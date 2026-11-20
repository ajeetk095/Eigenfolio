"""
Live + historical financial data integration (Environment B data feed).

Yahoo Finance is a best-effort public source, not a guaranteed uptime API.
The loader tolerates yfinance's changing MultiIndex layouts, retries missing
symbols individually, and never invents prices for unavailable observations.
"""
from __future__ import annotations

import hashlib
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import DATA_SOURCE_LABEL

_CACHE: dict = {}
CACHE_TTL_SECONDS = 600          # 10-minute TTL for live quotes
HIST_CACHE_TTL_SECONDS = 3600    # 1-hour TTL for historical windows


class DataFetchError(RuntimeError):
    """Raised when a market-data source fails or returns unusable data."""


def _cache_get(key: str, ttl: int):
    item = _CACHE.get(key)
    if item and (time.time() - item['t']) < ttl:
        return item['data']
    return None


def _cache_put(key: str, data):
    _CACHE[key] = {'data': data, 't': time.time()}


def _extract_close(raw: pd.DataFrame | None, tickers: list[str]) -> pd.DataFrame:
    """Normalize yfinance Close output across single/multi-ticker layouts."""
    if raw is None or raw.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=tickers, dtype=float)

    close = None
    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance may return either (Price, Ticker) or (Ticker, Price).
        for level in range(raw.columns.nlevels):
            if 'Close' in raw.columns.get_level_values(level):
                try:
                    close = raw.xs('Close', axis=1, level=level, drop_level=True)
                    break
                except (KeyError, ValueError):
                    continue
        if isinstance(close, pd.Series):
            close = close.to_frame()
        if close is None:
            # Defensive fallback for MultiIndex columns with unusual level names.
            selected = {}
            for ticker in tickers:
                for col in raw.columns:
                    parts = tuple(str(part) for part in col)
                    if ticker in parts and 'Close' in parts:
                        selected[ticker] = raw[col]
                        break
            close = pd.DataFrame(selected, index=raw.index)
    elif 'Close' in raw.columns:
        close = raw[['Close']].rename(columns={'Close': tickers[0]}) if len(tickers) == 1 else None
    elif len(tickers) == 1 and raw.shape[1] == 1:
        close = raw.rename(columns={raw.columns[0]: tickers[0]})
    else:
        # Accept an already extracted ticker-indexed close frame.
        close = raw.reindex(columns=tickers)

    if close is None:
        return pd.DataFrame(index=raw.index, columns=tickers, dtype=float)
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close.columns = [str(c) for c in close.columns]
    # A single-symbol download can have a plain Close column after extraction.
    if len(tickers) == 1 and tickers[0] not in close.columns and close.shape[1] == 1:
        close.columns = tickers
    return close.reindex(columns=tickers).apply(pd.to_numeric, errors='coerce')


def _download_history(tickers: list[str], start: str, end_exclusive: str) -> pd.DataFrame:
    """Bulk download, then recover missing ticker columns with individual requests."""
    import yfinance as yf

    try:
        raw = yf.download(
            tickers, start=start, end=end_exclusive, auto_adjust=True,
            progress=False, threads=False, timeout=20, group_by='column',
        )
        prices = _extract_close(raw, tickers)
    except Exception:
        prices = pd.DataFrame(columns=tickers, dtype=float)

    missing = [t for t in tickers if t not in prices or prices[t].notna().sum() == 0]
    for ticker in missing:
        try:
            raw_one = yf.download(
                ticker, start=start, end=end_exclusive, auto_adjust=True,
                progress=False, threads=False, timeout=20,
            )
            one = _extract_close(raw_one, [ticker])
            if one[ticker].notna().any():
                prices[ticker] = one[ticker]
        except Exception:
            continue
    return prices.reindex(columns=tickers).sort_index()


def fetch_price_history(tickers: list[str], start: str, end: str,
                        force_refresh: bool = False) -> dict:
    """Download adjusted daily closes; ``end`` is treated as inclusive.

    Returns ``{'prices': DataFrame, 'fetched_at': ISO timestamp, 'source': label}``.
    Raises DataFetchError with actionable context if the provider has no usable data.
    """
    key = 'hist-v2:' + hashlib.md5(
        f"{sorted(tickers)}|{start}|{end}".encode(),
        usedforsecurity=False).hexdigest()
    if not force_refresh:
        cached = _cache_get(key, HIST_CACHE_TTL_SECONDS)
        if cached is not None:
            return cached

    try:
        # Yahoo's end parameter is exclusive; callers and app config use inclusive dates.
        end_exclusive = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
        prices = _download_history(tickers, start, end_exclusive)
        prices = prices.loc[:, [t for t in tickers if t in prices.columns]]
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        prices = prices[~prices.index.duplicated(keep='last')].sort_index()
        prices = prices.dropna(how='all')
        missing = [t for t in tickers if t not in prices or prices[t].notna().sum() == 0]
        if prices.empty or missing:
            suffix = f" No usable history for: {', '.join(missing)}." if missing else ''
            raise DataFetchError(
                'Yahoo Finance returned no usable historical prices.' + suffix +
                ' Check the connection and retry; market-data access can be temporarily limited.')
        prices = prices.reindex(columns=tickers)
    except DataFetchError:
        raise
    except Exception as exc:                       # noqa: BLE001
        raise DataFetchError(
            f'Market-data download failed ({type(exc).__name__}). '
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
    """Return latest available delayed/end-of-day quotes from recent price history."""
    key = 'quotes-v2:' + hashlib.md5(str(sorted(tickers)).encode(),
                                    usedforsecurity=False).hexdigest()
    if not force_refresh:
        cached = _cache_get(key, CACHE_TTL_SECONDS)
        if cached is not None:
            return cached

    try:
        import yfinance as yf
    except Exception:                              # noqa: BLE001
        raise DataFetchError('yfinance is not installed in this environment.') from None

    try:
        raw = yf.download(tickers, period='10d', auto_adjust=True, progress=False,
                          threads=False, timeout=20, group_by='column')
        recent = _extract_close(raw, tickers)
    except Exception:
        recent = pd.DataFrame(columns=tickers, dtype=float)

    rows, errors = [], []
    for ticker in tickers:
        series = recent[ticker].dropna() if ticker in recent else pd.Series(dtype=float)
        last = float(series.iloc[-1]) if len(series) else None
        prev = float(series.iloc[-2]) if len(series) > 1 else None

        # fast_info provides a useful fallback where the historical endpoint omitted a symbol.
        if last is None:
            try:
                info = yf.Ticker(ticker).fast_info
                last = info.get('last_price')
                prev = info.get('previous_close')
                last = float(last) if last is not None else None
                prev = float(prev) if prev is not None else None
            except Exception:
                pass
        if last is None:
            errors.append(ticker)
            continue
        rows.append({
            'Ticker': ticker,
            'Last Price': last,
            'Prev Close': prev if prev is not None else np.nan,
            'Change %': (last / prev - 1.0) * 100 if prev else np.nan,
        })

    if not rows:
        raise DataFetchError(
            'Latest quotes are temporarily unavailable from Yahoo Finance. '
            'Historical prices may still be available; retry in a few minutes.')

    payload = {
        'quotes': pd.DataFrame(rows),
        'failed_tickers': errors,
        'fetched_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'source': DATA_SOURCE_LABEL,
    }
    _cache_put(key, payload)
    return payload
