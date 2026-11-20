"""
Forecasting utilities — clearly-labelled additional research components.

The original GWP3 model contains NO return-forecasting capability: it
produces expected-return *estimates* from historical means and a covariance
estimate via Ledoit-Wolf. These wrappers expose exactly that (with
uncertainty and horizon disclosure) and nothing more. No fabricated
AI predictions are generated anywhere in this project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RF_ANNUAL, TRADING_DAYS


def expected_return_estimate(train_returns: pd.DataFrame) -> pd.DataFrame:
    """Historical-mean expected-return estimates with a 1-sigma uncertainty band.

    Methodology: annualised mean of daily log returns over the training window.
    Uncertainty: standard error of the mean, x sqrt(252).
    """
    n = len(train_returns)
    mu_ann = train_returns.mean() * TRADING_DAYS
    se_ann = train_returns.std() / np.sqrt(n) * TRADING_DAYS
    return pd.DataFrame({
        'Expected Return (ann.)': mu_ann,
        'Uncertainty (±1 SE)': se_ann,
        'Lower': mu_ann - se_ann,
        'Upper': mu_ann + se_ann,
        'Training Days': n,
        'Method': 'Historical mean of daily log returns (no predictive model)',
        'Horizon': 'Annualised estimate, not a forecast',
    })


def volatility_forecast(train_returns: pd.DataFrame,
                        denoise_info: dict) -> pd.DataFrame:
    """Forward volatility estimate = sqrt(diagonal of LW-shrunk covariance).

    Uses the most recent training-window LW covariance as the simplest
    unbiased-style volatility estimate; disclosed as an estimate, not a
    time-varying forecast (no GARCH-type dynamics in the original model).
    """
    S_lw = denoise_info['S_lw']
    return pd.DataFrame({
        'Ann. Vol Estimate': np.sqrt(np.diag(S_lw.values)),
        'Ann. Vol (sample)': train_returns.std() * np.sqrt(TRADING_DAYS),
        'Method': 'sqrt(diag(Ledoit-Wolf shrunk covariance))',
        'Horizon': 'Next-period estimate (constant)',
    })


def excess_return_scan(mu: pd.Series, S: pd.DataFrame) -> pd.DataFrame:
    """Assets whose estimated return exceeds the risk-free rate — a screening
    output of the existing model, clearly labelled as an estimate."""
    df = pd.DataFrame({
        'Est. Ann. Return': mu,
        'Est. Ann. Vol': np.sqrt(np.diag(S.values)),
        'Excess over rf (5.25%)': mu - RF_ANNUAL,
    })
    df['Above rf?'] = df['Excess over rf (5.25%)'] > 0
    return df.sort_values('Excess over rf (5.25%)', ascending=False)
