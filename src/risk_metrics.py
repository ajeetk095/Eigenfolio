"""
Portfolio performance and risk metrics.

Conventions (identical to the original GWP3 notebook, disclosed in the report):
- Daily log returns; annualisation uses 252 trading days.
- Ann. Return  = mean daily log return x 252
- Ann. Vol     = std daily log return x sqrt(252)
- Sharpe       = mean daily excess / std daily excess x sqrt(252), rf = 5.25% p.a.
- Sortino      = (AnnRet - rf) / annualised downside deviation
- Max Drawdown = min of wealth index (exp of cumulative log return) vs its running max
- CVaR 95%     = mean of the worst 5% of daily returns, annualised by sqrt(252)
                 (as implemented in the original notebook)
- Calmar       = Ann. Return / |Max Drawdown|
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RF_ANNUAL, RF_DAILY, TRADING_DAYS

METRIC_KEYS = ['Ann. Return', 'Ann. Vol', 'Sharpe', 'Sortino',
               'Max Drawdown', 'CVaR 95%', 'Calmar']


def portfolio_metrics(weights, returns: pd.DataFrame, rf_daily: float = RF_DAILY) -> dict:
    """Compute the full annualised metric set for fixed portfolio weights.

    weights : array-like of length n_assets (must sum to ~1)
    returns : DataFrame of daily log returns, shape (T, n_assets)
    """
    w = np.asarray(weights, dtype=float)
    daily_port = returns.values @ w

    ann_ret = daily_port.mean() * TRADING_DAYS
    ann_vol = daily_port.std() * np.sqrt(TRADING_DAYS)

    excess = daily_port - rf_daily
    sharpe = (excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)
              if excess.std() > 0 else 0.0)

    downside = daily_port[daily_port < rf_daily]
    sortino = ((ann_ret - RF_ANNUAL) / (downside.std() * np.sqrt(TRADING_DAYS))
               if len(downside) > 1 and downside.std() > 0 else 0.0)

    cum_ret = np.exp(np.cumsum(daily_port))
    roll_max = np.maximum.accumulate(cum_ret)
    max_dd = ((cum_ret - roll_max) / roll_max).min()

    var_95 = np.percentile(daily_port, 5)
    cvar_95 = daily_port[daily_port <= var_95].mean() * np.sqrt(TRADING_DAYS)

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    return {'Ann. Return': ann_ret, 'Ann. Vol': ann_vol, 'Sharpe': sharpe,
            'Sortino': sortino, 'Max Drawdown': max_dd, 'CVaR 95%': cvar_95,
            'Calmar': calmar}


def daily_var_es(daily_port_returns: np.ndarray, alpha: float = 0.05) -> dict:
    """Historical daily VaR and Expected Shortfall at a given confidence level.

    Returns daily (non-annualised) figures with clearly stated conventions.
    """
    r = np.asarray(daily_port_returns, dtype=float)
    var = float(np.percentile(r, 100 * alpha))
    es = float(r[r <= var].mean()) if np.any(r <= var) else var
    return {'VaR': var, 'ES': es, 'confidence': 1 - alpha}


def cumulative_returns(weights, returns: pd.DataFrame) -> pd.Series:
    """Daily wealth index (starts at 1) from daily log returns."""
    dp = returns.values @ np.asarray(weights, dtype=float)
    return pd.Series(np.exp(np.cumsum(dp)), index=returns.index)


def drawdown_series(weights, returns: pd.DataFrame) -> pd.Series:
    """Drawdown series (<= 0) of the wealth index."""
    cum = cumulative_returns(weights, returns)
    roll_max = cum.cummax()
    return (cum - roll_max) / roll_max


def risk_contributions(weights, cov: pd.DataFrame) -> pd.Series:
    """Percent contribution to portfolio variance per asset.

    RC_i = w_i * (Sigma w)_i / (w' Sigma w), normalised to sum to 100%.
    """
    w = np.asarray(weights, dtype=float)
    S = cov.values
    port_var = float(w @ S @ w)
    if port_var <= 0:
        return pd.Series(0.0, index=cov.index)
    marginal = S @ w
    contrib = w * marginal / port_var
    return pd.Series(contrib, index=cov.index)


def concentration_hhi(weights) -> float:
    """Herfindahl-Hirschman Index of portfolio weights (0..1; 1 = single asset)."""
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w ** 2))


def effective_n(weights) -> float:
    """Effective number of independent positions = 1 / HHI."""
    hhi = concentration_hhi(weights)
    return float(1.0 / hhi) if hhi > 0 else 0.0


def validate_weights(weights, max_weight: float = 0.15,
                     tol: float = 1e-6) -> dict:
    """Validate the assignment constraints on a weight vector."""
    w = np.asarray(weights, dtype=float)
    return {
        'sum_to_one': bool(abs(w.sum() - 1.0) <= 1e-4),
        'no_shorts': bool((w >= -tol).all()),
        'cap_respected': bool((w <= max_weight + tol).all()),
        'sum': float(w.sum()),
        'max_weight': float(w.max()),
        'min_weight': float(w.min()),
    }
