"""
Step 1 / Step 3 — Portfolio optimisation.

Primary path: PyPortfolioOpt EfficientFrontier.max_sharpe with
weight_bounds=(0, 0.15) — exactly as in the original GWP3 notebook.
Fallback path: a SciPy SLSQP maximum-Sharpe optimiser with identical
constraints, used only if PyPortfolioOpt/cvxpy is unavailable in the
deployment environment. Both are max-Sharpe MVO; constraints are identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import MAX_WEIGHT, RF_ANNUAL


def _scipy_max_sharpe(mu: pd.Series, S: pd.DataFrame,
                      max_weight: float = MAX_WEIGHT) -> np.ndarray:
    """SciPy fallback: maximise (w'mu - rf) / sqrt(w'Sw), long-only, capped."""
    n = len(mu)
    rf = RF_ANNUAL

    def neg_sharpe(w):
        ret = w @ mu.values
        vol = np.sqrt(max(w @ S.values @ w, 1e-12))
        return -(ret - rf) / vol

    res = minimize(
        neg_sharpe,
        x0=np.full(n, 1.0 / n),
        bounds=[(0.0, max_weight)] * n,
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}],
        method='SLSQP',
        options={'maxiter': 1000, 'ftol': 1e-12},
    )
    w = np.clip(res.x, 0, max_weight)
    return w / w.sum()


def max_sharpe_weights(mu: pd.Series, S: pd.DataFrame,
                       max_weight: float = MAX_WEIGHT,
                       risk_free_rate: float = RF_ANNUAL) -> np.ndarray:
    """Maximum-Sharpe MVO weights, long-only with per-asset cap.

    Uses PyPortfolioOpt (as in the original notebook) when available.
    mu  : annualised expected returns (index = tickers)
    S   : annualised covariance matrix (DataFrame, index/cols = tickers)
    """
    tickers = list(mu.index)
    try:
        from pypfopt import EfficientFrontier
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        wd = ef.clean_weights()
        return np.array([wd[t] for t in tickers])
    except Exception:
        return _scipy_max_sharpe(mu, S, max_weight)


def efficient_frontier(mu: pd.Series, S: pd.DataFrame,
                       max_weight: float = MAX_WEIGHT,
                       n_points: int = 60) -> pd.DataFrame:
    """Trace the constrained efficient frontier (vol, return) points.

    Mirrors the notebook's frontier sweep (300 targets there; 60 is enough
    for an interactive plot and much faster in an app context).
    """
    target_rets = np.linspace(float(mu.min()), float(mu.max()) * 0.95, n_points)
    pts = []
    n = len(mu)
    for tr in target_rets:
        try:
            from pypfopt import EfficientFrontier
            ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
            ef.efficient_return(tr)
            wd = ef.clean_weights()
            w = np.array([wd[t] for t in mu.index])
            vol = float(np.sqrt(w @ S.values @ w))
            pts.append({'vol': vol, 'ret': tr})
        except Exception:
            # SciPy fallback for this target: minimise vol s.t. w'mu >= tr
            try:
                res = minimize(
                    lambda w: w @ S.values @ w,
                    x0=np.full(n, 1.0 / n),
                    bounds=[(0.0, max_weight)] * n,
                    constraints=[
                        {'type': 'eq', 'fun': lambda w: w.sum() - 1.0},
                        {'type': 'ineq', 'fun': lambda w, t=tr: w @ mu.values - t},
                    ],
                    method='SLSQP',
                )
                if res.success:
                    w = np.clip(res.x, 0, max_weight)
                    w /= w.sum()
                    pts.append({'vol': float(np.sqrt(w @ S.values @ w)),
                                'ret': float(w @ mu.values)})
            except Exception:
                continue
    return pd.DataFrame(pts)


def enforce_cap_iterative(w: np.ndarray, cap: float = MAX_WEIGHT,
                          max_iter: int = 50) -> np.ndarray:
    """Enforce the per-asset cap while keeping weights summing to 1.

    NOTE ON A DISCLOSED DISCREPANCY: the original notebook used a single
    clip-then-renormalise pass, which can push weights BACK above the cap
    after renormalisation (the submitted report's Table 10 therefore shows
    COST at 24.14% despite the 15% rule). This app enforces the assignment
    cap correctly with iterative capping and discloses the difference on the
    Optimization and Methodology pages.
    """
    w = np.clip(np.asarray(w, dtype=float), 0.0, cap)
    for _ in range(max_iter):
        total = w.sum()
        if total <= 0:
            return np.full_like(w, 1.0 / len(w))
        w = w / total
        over = w > cap + 1e-9
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w > 0)
        if free.any() and excess > 0:
            w[free] += excess * w[free] / w[free].sum()
        else:
            w[~over] += excess / max((~over).sum(), 1)
    return w


def combination_portfolio(train_returns: pd.DataFrame, tickers: list[str],
                          labels: np.ndarray, n_clusters: int,
                          intra_weights: dict) -> dict:
    """Step 3 combination portfolio: Denoise -> Cluster -> MVO.

    Exactly the original notebook pipeline:
    A. cluster-level return series (inverse-vol intra-cluster weights)
    B. Ledoit-Wolf on cluster returns, annualised x252
    C. cluster-level max-Sharpe MVO (bounds 0..1)
    D. expand to assets, apply 15% cap, renormalise
    """
    from sklearn.covariance import LedoitWolf
    from .config import TRADING_DAYS

    cluster_ret = pd.DataFrame({
        f'C{c + 1}': train_returns.iloc[:, intra_weights[c][0]].values
                     @ intra_weights[c][1]
        for c in range(n_clusters)
    })

    lw = LedoitWolf().fit(cluster_ret.values)
    S_combo = pd.DataFrame(lw.covariance_ * TRADING_DAYS,
                           index=cluster_ret.columns, columns=cluster_ret.columns)
    mu_combo = pd.Series(cluster_ret.mean().values * TRADING_DAYS,
                         index=cluster_ret.columns)

    ef_cluster_w = max_sharpe_weights(mu_combo, S_combo, max_weight=1.0)
    cluster_alloc = {f'C{c + 1}': float(ef_cluster_w[c]) for c in range(n_clusters)}

    combo = np.zeros(len(tickers))
    for c in range(n_clusters):
        idx, intra_w = intra_weights[c]
        combo[idx] = intra_w * cluster_alloc[f'C{c + 1}']

    combo = enforce_cap_iterative(combo)

    return {'weights': combo, 'cluster_alloc': cluster_alloc,
            'mu_cluster': mu_combo, 'S_cluster': S_combo}
