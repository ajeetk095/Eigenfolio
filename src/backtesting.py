"""
Step 2/3 ML improvement — Backtesting via rolling walk-forward.

Preserved from the original GWP3 notebook:
- window W = 126 trading days (~6 months), step S = 21 days (~1 month)
- Ledoit-Wolf fitted on each in-sample window; expected returns =
  annualised mean of in-sample log returns
- max-Sharpe MVO with (0, 0.15) bounds; equal-weight fallback on failure
- only the TRAINING period is used (no test-data leakage)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BT_STEP, BT_WINDOW, MAX_WEIGHT, RF_ANNUAL, TRADING_DAYS
from .risk_metrics import portfolio_metrics


def rolling_walk_forward(log_returns: pd.DataFrame, tickers: list[str],
                         window: int = BT_WINDOW, step: int = BT_STEP,
                         train_end: str | None = None) -> dict:
    """Run the rolling walk-forward backtest on the training period.

    Returns a per-window metrics DataFrame, the concatenated OOS daily
    return series and the cumulative OOS wealth index.
    """
    from sklearn.covariance import LedoitWolf

    bt = log_returns[log_returns.index <= train_end] if train_end else log_returns

    records, oos_daily, oos_dates = [], [], []
    n_days = len(bt)
    starts = range(0, n_days - window - step, step)

    for s in starts:
        in_r = bt.iloc[s: s + window]
        out_r = bt.iloc[s + window: s + window + step]
        if len(out_r) < 1:
            continue

        lw_bt = LedoitWolf().fit(in_r.values)
        S_bt = pd.DataFrame(lw_bt.covariance_ * TRADING_DAYS,
                            index=tickers, columns=tickers)
        mu_bt = pd.Series(in_r.mean().values * TRADING_DAYS, index=tickers)

        try:
            from pypfopt import EfficientFrontier
            ef = EfficientFrontier(mu_bt, S_bt, weight_bounds=(0, MAX_WEIGHT))
            ef.max_sharpe(risk_free_rate=RF_ANNUAL)
            wd = ef.clean_weights()
            w_bt = np.array([wd[t] for t in tickers])
        except Exception:
            # Equal-weight fallback if the optimiser fails (near-singular S)
            w_bt = np.ones(len(tickers)) / len(tickers)

        oos_dp = out_r.values @ w_bt
        oos_daily.extend(oos_dp.tolist())
        oos_dates.extend(out_r.index.tolist())

        m = portfolio_metrics(w_bt, out_r)
        records.append({
            'window_start': in_r.index[0].date(),
            'window_end': in_r.index[-1].date(),
            'oos_start': out_r.index[0].date(),
            'oos_end': out_r.index[-1].date(),
            **m,
        })

    bt_df = pd.DataFrame(records)
    oos_series = pd.Series(oos_daily,
                           index=pd.DatetimeIndex(oos_dates)).sort_index()
    cum_oos = pd.Series(np.exp(np.cumsum(oos_series.values)), index=oos_series.index)

    return {'windows': bt_df, 'oos_series': oos_series, 'cum_oos': cum_oos}
