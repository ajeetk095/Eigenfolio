"""
Step 2 ML improvement — Denoising via Ledoit-Wolf covariance shrinkage.

Methodology preserved exactly from the original GWP3 notebook:
sklearn.covariance.LedoitWolf fitted on daily training log returns,
covariance annualised by x252. Expected returns are NOT shrunk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import TRADING_DAYS


def lw_shrinkage(train_returns: pd.DataFrame, tickers: list[str]) -> dict:
    """Fit Ledoit-Wolf on training daily log returns.

    Returns the shrunk annualised covariance, shrinkage intensity,
    eigenvalue spectra and condition numbers for both estimators.
    """
    lw = LedoitWolf().fit(train_returns.values)
    alpha = float(lw.shrinkage_)

    S_lw = pd.DataFrame(lw.covariance_ * TRADING_DAYS,
                        index=tickers, columns=tickers)

    # Sample covariance computed on the same daily log returns (annualised)
    sample_daily = np.cov(train_returns.values, rowvar=False)
    S_sample = pd.DataFrame(sample_daily * TRADING_DAYS,
                            index=tickers, columns=tickers)

    eig_sample = np.sort(np.linalg.eigvalsh(S_sample.values))[::-1]
    eig_lw = np.sort(np.linalg.eigvalsh(S_lw.values))[::-1]

    cn_sample = float(np.linalg.cond(S_sample.values))
    cn_lw = float(np.linalg.cond(S_lw.values))

    corr_lw = pd.DataFrame(
        lw.covariance_ / np.sqrt(np.outer(np.diag(lw.covariance_),
                                          np.diag(lw.covariance_))),
        index=tickers, columns=tickers)

    return {
        'S_lw': S_lw,
        'S_sample': S_sample,
        'alpha': alpha,
        'eig_sample': eig_sample,
        'eig_lw': eig_lw,
        'cond_sample': cn_sample,
        'cond_lw': cn_lw,
        'cond_improvement': cn_sample / cn_lw if cn_lw > 0 else np.nan,
        'corr_sample': train_returns.corr(),
        'corr_lw': corr_lw,
    }
