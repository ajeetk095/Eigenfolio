"""
Data validation and preprocessing — same procedures as the original notebook:
- reindex columns to the fixed TICKERS order
- forward-fill sporadic missing prices, drop remaining NaNs
- daily log returns; simple returns for PyPortfolioOpt inputs
- duplicate-index check, alignment check
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TEST_END, TEST_START, TRAIN_END, TRAIN_START


def clean_prices(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Notebook-identical cleaning: reindex, ffill, dropna."""
    prices = prices.reindex(columns=tickers)
    prices = prices.ffill().dropna(how='all')
    prices = prices.ffill().dropna()
    return prices


def compute_returns(prices: pd.DataFrame) -> dict:
    """Daily log returns (for metrics) and simple returns (for pypfopt)."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    simple_returns = prices.pct_change().dropna()
    return {'log': log_returns, 'simple': simple_returns}


def split_periods(returns: pd.DataFrame) -> dict:
    """Split a return frame into the academic train / OOS test windows."""
    train = returns[(returns.index >= TRAIN_START) & (returns.index <= TRAIN_END)]
    test = returns[(returns.index >= TEST_START) & (returns.index <= TEST_END)]
    return {'train': train, 'test': test}


def validate_data(prices: pd.DataFrame, log_returns: pd.DataFrame) -> dict:
    """Run integrity checks and return a pass/fail report for display."""
    dupes = int(prices.index.duplicated().sum())
    n_missing_prices = int(prices.isna().sum().sum())
    n_missing_returns = int(log_returns.isna().sum().sum())
    non_finite = int((~np.isfinite(log_returns.values)).sum())
    aligned = list(prices.columns) == list(log_returns.columns)
    ok = (dupes == 0 and n_missing_prices == 0 and n_missing_returns == 0
          and non_finite == 0 and aligned)
    return {
        'ok': ok,
        'duplicate_dates': dupes,
        'missing_prices': n_missing_prices,
        'missing_returns': n_missing_returns,
        'non_finite_returns': non_finite,
        'columns_aligned': aligned,
        'n_days': len(log_returns),
        'first_date': str(log_returns.index[0].date()) if len(log_returns) else '—',
        'last_date': str(log_returns.index[-1].date()) if len(log_returns) else '—',
    }
