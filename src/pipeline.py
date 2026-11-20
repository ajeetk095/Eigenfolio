"""
Reusable pipeline for the GWP3 model — Environment A (academic validation).

The assignment's fixed windows are validated by date and observation count before
optimization, so an incomplete Yahoo Finance response cannot silently produce
misleading portfolio outputs.
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
    """Run the full model on validated downloaded data and cache the result."""
    key = hashlib.md5(f'full-v2|{run_backtest}'.encode(),
                      usedforsecurity=False).hexdigest()
    if not force_refresh and key in _RESULT_CACHE:
        return _RESULT_CACHE[key]

    hist = data_loader.fetch_price_history(
        TICKERS, config.TRAIN_START, config.TEST_END, force_refresh=force_refresh)
    prices = data_validation.clean_prices(hist['prices'], TICKERS)
    if prices.empty:
        raise data_loader.DataFetchError(
            'No complete price rows were returned for the academic period. '
            'Retry after the data provider recovers.')
    returns = data_validation.compute_returns(prices)
    log_ret = returns['log']
    periods = data_validation.split_periods(log_ret)
    train_ret, test_ret = periods['train'], periods['test']

    missing = [ticker for ticker in TICKERS
               if ticker not in prices or prices[ticker].notna().sum() < 300]
    if missing:
        raise data_loader.DataFetchError(
            'Historical coverage is incomplete for: ' + ', '.join(missing) +
            '. Yahoo Finance may have returned partial data; retry later.')
    if len(train_ret) < 300 or len(test_ret) < 15:
        train_range = (f"{train_ret.index.min().date()}–{train_ret.index.max().date()}"
                       if len(train_ret) else 'no training dates')
        test_range = (f"{test_ret.index.min().date()}–{test_ret.index.max().date()}"
                      if len(test_ret) else 'no test dates')
        raise data_loader.DataFetchError(
            'The downloaded history does not adequately cover the fixed academic '
            f'windows (training: {train_range}, {len(train_ret)} days; '
            f'July 2025 test: {test_range}, {len(test_ret)} days). '
            'Check that Yahoo Finance is reachable and retry.')

    validation = data_validation.validate_data(prices, log_ret)
    if not validation['ok']:
        raise data_loader.DataFetchError(
            'Downloaded prices failed data-integrity checks. Retry later; '
            'incomplete or inconsistent market data is not used for optimization.')

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
    _, intra_w = clustering.cluster_return_series(train_ret, cl['labels'], TICKERS)
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
        winners[col] = table[col].idxmin() if col == 'Ann. Vol' else table[col].idxmax()
    return winners
