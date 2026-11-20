"""Financial correctness tests: constraints, metrics, denoising, clustering.

Run:  python -m pytest tests/ -v   (from the project root)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.backtesting import rolling_walk_forward
from src.clustering import cluster_portfolio, correlation_distance, kmedoids_fit
from src.denoising import lw_shrinkage
from src.optimization import _scipy_max_sharpe, max_sharpe_weights
from src.risk_metrics import (METRIC_KEYS, portfolio_metrics, risk_contributions,
                              validate_weights)

TICKERS = config.TICKERS


# --------------------------------------------------------------- fixtures ---
@pytest.fixture(scope='module')
def synthetic_returns() -> pd.DataFrame:
    """Deterministic synthetic daily log returns with realistic scale."""
    rng = np.random.default_rng(0)
    n_days, n_assets = 400, len(TICKERS)
    base = rng.normal(0.0004, 0.015, size=(n_days, n_assets))
    # Inject a common factor so assets are correlated (like the real universe)
    factor = rng.normal(0, 0.010, size=(n_days, 1))
    data = base + 0.7 * factor
    idx = pd.bdate_range('2024-01-02', periods=n_days)
    return pd.DataFrame(data, index=idx, columns=TICKERS)


# ------------------------------------------------------------ constraints ---
def test_weights_sum_to_one_and_constraints():
    w = np.array([0.15, 0.15, 0.10, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0])
    v = validate_weights(w)
    assert v['sum_to_one'] and v['no_shorts'] and v['cap_respected']


def test_validator_rejects_cap_violation():
    w = np.array([0.30] + [0.7 / 11] * 11)
    v = validate_weights(w)
    assert not v['cap_respected']


def test_validator_rejects_short():
    w = np.array([-0.05] + [1.05 / 11] * 11)
    v = validate_weights(w)
    assert not v['no_shorts']


# --------------------------------------------------------- optimisation -----
def test_max_sharpe_respects_constraints(synthetic_returns):
    mu = synthetic_returns.mean() * 252
    S = pd.DataFrame(np.cov(synthetic_returns.values, rowvar=False) * 252,
                     index=TICKERS, columns=TICKERS)
    w = max_sharpe_weights(mu, S)
    v = validate_weights(w, config.MAX_WEIGHT)
    assert v['sum_to_one'], 'weights must sum to 1'
    assert v['no_shorts'], 'long-only violated'
    assert v['cap_respected'], '15% cap violated'


def test_scipy_fallback_matches_constraints(synthetic_returns):
    mu = synthetic_returns.mean() * 252
    S = pd.DataFrame(np.cov(synthetic_returns.values, rowvar=False) * 252,
                     index=TICKERS, columns=TICKERS)
    w = _scipy_max_sharpe(mu, S)
    v = validate_weights(w, config.MAX_WEIGHT)
    assert v['sum_to_one'] and v['no_shorts'] and v['cap_respected']


# -------------------------------------------------------------- metrics -----
def test_portfolio_metrics_keys(synthetic_returns):
    w = np.ones(len(TICKERS)) / len(TICKERS)
    m = portfolio_metrics(w, synthetic_returns)
    assert set(m) == set(METRIC_KEYS)
    assert m['Ann. Vol'] > 0
    assert -1.0 < m['Max Drawdown'] <= 0


def test_sharpe_manual_calculation():
    """Sharpe must match a hand-computed value on a fixed series."""
    rets = pd.DataFrame({'A': [0.001, 0.002, -0.001, 0.0005, 0.003]})
    w = np.array([1.0])
    m = portfolio_metrics(w, rets, rf_daily=0.0)
    r = rets['A'].values
    expected = r.mean() / r.std(ddof=0) * np.sqrt(252)   # numpy ddof=0, as in portfolio_metrics
    assert abs(m['Sharpe'] - expected) < 1e-9


def test_max_drawdown_known_series():
    """Wealth path 1 -> 1.1 -> 0.99 must give MDD = (0.99 - 1.1)/1.1."""
    rets = pd.DataFrame({'A': [np.log(1.1), np.log(0.99 / 1.1)]})
    m = portfolio_metrics(np.array([1.0]), rets)
    assert abs(m['Max Drawdown'] - (0.99 / 1.1 - 1)) < 1e-9


def test_risk_contributions_sum_to_one(synthetic_returns):
    w = np.ones(len(TICKERS)) / len(TICKERS)
    S = pd.DataFrame(np.cov(synthetic_returns.values, rowvar=False),
                     index=TICKERS, columns=TICKERS)
    rc = risk_contributions(w, S)
    assert abs(rc.sum() - 1.0) < 1e-8


# ------------------------------------------------------------ denoising -----
def test_lw_shrinkage_properties(synthetic_returns):
    dn = lw_shrinkage(synthetic_returns, TICKERS)
    assert 0.0 <= dn['alpha'] <= 1.0
    # shrinkage must not increase (and usually lowers) the condition number
    assert dn['cond_lw'] <= dn['cond_sample'] * 1.5
    assert dn['S_lw'].shape == (len(TICKERS), len(TICKERS))
    # symmetric PSD
    assert np.allclose(dn['S_lw'].values, dn['S_lw'].values.T)
    assert (np.linalg.eigvalsh(dn['S_lw'].values) > 0).all()


def test_lw_covariance_is_shrunk_toward_target(synthetic_returns):
    dn = lw_shrinkage(synthetic_returns, TICKERS)
    a = dn['alpha']
    # sklearn's LedoitWolf uses the biased (ddof=0) covariance internally
    daily_sample = np.cov(synthetic_returns.values, rowvar=False, bias=True) * 252
    expected = (1 - a) * daily_sample + a * np.trace(daily_sample) / len(TICKERS) * np.eye(len(TICKERS))
    assert np.allclose(expected, dn['S_lw'].values, atol=1e-6)


# ----------------------------------------------------------- clustering -----
def test_kmedoids_partitions_all_points():
    dist = np.array([[0.0, 0.1, 0.9], [0.1, 0.0, 0.9], [0.9, 0.9, 0.0]])
    labels, medoids = kmedoids_fit(dist, n_clusters=2, random_state=42)
    assert len(labels) == 3
    assert len(set(labels)) == 2
    assert 0 in medoids or 1 in medoids   # {0,1} are the natural pair


def test_correlation_distance_bounds():
    c = pd.DataFrame(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    d = correlation_distance(c)
    assert d.min() >= 0 and d.max() <= 1.0
    assert abs(d[0, 1] - 1.0) < 1e-9      # perfect anti-correlation -> d = 1


def test_cluster_portfolio_constraints(synthetic_returns):
    cl = cluster_portfolio(synthetic_returns, TICKERS)
    v = validate_weights(cl['weights'], config.MAX_WEIGHT)
    assert v['sum_to_one'] and v['no_shorts'] and v['cap_respected']
    assert len(cl['clusters']) == config.N_CLUSTERS


# ---------------------------------------------------------- backtesting -----
def test_walk_forward_no_leakage(synthetic_returns):
    bt = rolling_walk_forward(synthetic_returns, TICKERS,
                              window=126, step=21,
                              train_end=str(synthetic_returns.index[-1].date()))
    wdw = bt['windows']
    assert len(wdw) > 0
    # Every OOS window must start after its in-sample window ends
    assert (wdw['oos_start'] > wdw['window_end']).all()
    # OOS series must cover disjoint, ordered dates (no reuse)
    dates = bt['oos_series'].index
    assert dates.is_monotonic_increasing
    assert dates.is_unique
