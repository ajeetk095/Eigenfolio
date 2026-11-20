"""
Reusable pipeline for the GWP3 model — Environment A (academic validation).

Runs the exact original methodology once and caches the result in memory:
1. baseline max-Sharpe MVO (sample covariance, bounds (0, 0.15))
2. Ledoit-Wolf denoising diagnostics + denoised portfolio
3. K-Medoids clustering (K=4) + pure cluster portfolio
4. combination portfolio (Denoise -> Cluster -> MVO)
5. rolling walk-forward backtest (W=126, S=21)
6. full metric tables for train and OOS periods

Static reference figures from the submitted report are kept in config and
never mixed with freshly computed outputs.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from . import backtesting, clustering, config, data_loader, data_validation
from .config import TICKERS
from .denoising import lw_shrinkage
from .optimization import combination_portfolio, efficient_frontier, max_sharpe_weights
from .risk_metrics import METRIC_KEYS, portfolio_metrics

_RESULT_CACHE: dict = {}


def build_full_result(force_refresh: bool = False,
                      run_backtest: bool = True) -> dict:
    """Run the entire GWP3 model on freshly downloaded data and cache it."""
    key = hashlib.md5(f'full|{run_backtest}'.encode(),
                      usedforsecurity=False).hexdigest()
    if not force_refresh and key in _RESULT_CACHE:
        return _RESULT_CACHE[key]

    hist = data_loader.fetch_price_history(
        TICKERS, config.TRAIN_START, config.TEST_END, force_refresh=force_refresh)
    prices = data_validation.clean_prices(hist['prices'], TICKERS)
    returns = data_validation.compute_returns(prices)
    log_ret = returns['log']
    periods = data_validation.split_periods(log_ret)
    train_ret, test_ret = periods['train'], periods['test']

    if len(train_ret) < 60 or len(test_ret) < 5:
        raise data_loader.DataFetchError(
            'Downloaded data does not cover the academic train/test windows '
            '(need Jan 2024 - Jul 2025). The provider may have returned '
            'incomplete data — please retry.')

    validation = data_validation.validate_data(prices, log_ret)

    # ---- Step 1: baseline MVO -------------------------------------------
    mu = train_ret.mean().values * config.TRADING_DAYS
    S_sample = np.cov(train_ret.values, rowvar=False) * config.TRADING_DAYS
    mu_s = pd.Series(mu, index=TICKERS)
    S_s = pd.DataFrame(S_sample, index=TICKERS, columns=TICKERS)
    base_w = max_sharpe_weights(mu_s, S_s)

    # ---- Step 2/3: denoising ---------------------------------------------
    dn = lw_shrinkage(train_ret, TICKERS)
    den_w = max_sharpe_weights(mu_s, dn['S_lw'])

    # ---- Step 2/3: clustering --------------------------------------------
    cl = clustering.cluster_portfolio(train_ret, TICKERS)
    clust_w = cl['weights']

    # ---- Step 3: combination ---------------------------------------------
    cl_ret, intra_w = clustering.cluster_return_series(
        train_ret, cl['labels'], TICKERS)
    combo = combination_portfolio(train_ret, TICKERS, cl['labels'],
                                  config.N_CLUSTERS, intra_w)

    # ---- Metrics ----------------------------------------------------------
    methods = {
        'Baseline (MVO)': base_w,
        'Denoise (LW)': den_w,
        'Cluster (K-Med)': clust_w,
        'Combo (Den+Clust)': combo['weights'],
    }
    metrics = {
        name: {'train': portfolio_metrics(w, train_ret),
               'test': portfolio_metrics(w, test_ret)}
        for name, w in methods.items()
    }

    # ---- Rolling backtest (training period only) ---------------------------
    bt = (backtesting.rolling_walk_forward(log_ret, TICKERS, train_end=config.TRAIN_END)
          if run_backtest else None)

    frontier = efficient_frontier(mu_s, S_s)

    result = {
        'prices': prices,
        'log_returns': log_ret,
        'train_ret': train_ret,
        'test_ret': test_ret,
        'validation': validation,
        'data_source': hist['source'],
        'fetched_at': hist['fetched_at'],
        'mu': mu_s,
        'S_sample': S_s,
        'weights': methods,
        'metrics': metrics,
        'denoise': dn,
        'clusters': cl,
        'combo': combo,
        'backtest': bt,
        'frontier': frontier,
        'constraint_checks': {name: validate_constraints(w) for name, w in methods.items()},
    }
    _RESULT_CACHE[key] = result
    return result


def validate_constraints(w: np.ndarray, cap: float = config.MAX_WEIGHT) -> dict:
    """Assignment constraint check: sums to 1, long-only, max 15% cap."""
    w = np.asarray(w, dtype=float)
    return {
        'sum_to_one': bool(abs(w.sum() - 1.0) <= 1e-4),
        'no_shorts': bool((w >= -1e-8).all()),
        'cap_respected': bool((w <= cap + 1e-6).all()),
        'max_weight': float(w.max()),
        'sum': float(w.sum()),
    }


def metrics_table(metrics: dict, period: str) -> pd.DataFrame:
    """Metric comparison table (methods x 7 metrics) for train or test."""
    rows = []
    for name, m in metrics.items():
        rows.append([name] + [m[period][k] for k in METRIC_KEYS])
    return pd.DataFrame(rows, columns=['Method'] + METRIC_KEYS).set_index('Method')


def winners_per_metric(table: pd.DataFrame) -> dict:
    """Best method per metric (less-negative drawdown/CVaR counts as better)."""
    winners = {}
    for col in table.columns:
        if col in ('Ann. Vol',):                       # lower is better
            winners[col] = table[col].idxmin()
        else:                                          # higher is better
            winners[col] = table[col].idxmax()
    return winners
