"""
Shared Streamlit utilities: model bootstrap, UI helpers, glossary and page footer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.pipeline import build_full_result, metrics_table, winners_per_metric  # noqa: E402


@st.cache_resource(show_spinner='Running the full GWP3 model on market data…')
def get_result(force_refresh: bool = False):
    """Compute and cache the complete GWP3 result from current provider data."""
    return build_full_result(force_refresh=force_refresh)


def weights_dict(result, method: str) -> dict:
    """Ticker-to-weight mapping for display."""
    return {t: float(w) for t, w in zip(config.TICKERS, result['weights'][method])}


def metric_tooltip(name: str, definition: str, why: str, how: str,
                   limit: str) -> None:
    st.markdown(f'**{name}**')
    with st.expander('What does this mean?'):
        st.markdown(f'- **Definition:** {definition}\n- **Why it matters:** {why}\n'
                    f'- **How to read it:** {how}\n- **Limitations:** {limit}')


def data_freshness_bar(result) -> None:
    """Show data provenance for model or history payloads, accepting either schema."""
    source = result.get('data_source') or result.get('source') or 'Unknown source'
    fetched_at = result.get('fetched_at') or 'unknown'
    validation = result.get('validation') or {}
    first_date = validation.get('first_date', '—')
    last_date = validation.get('last_date', '—')
    n_days = validation.get('n_days', '—')
    st.caption(f'Data source: {source} · Retrieved: {fetched_at} (UTC) · '
               f'Coverage: {first_date} → {last_date} ({n_days} trading days). '
               'Delayed / end-of-day data — not tick-by-tick real-time.')


def disclaimer_box() -> None:
    """Render the standard disclosure in the page footer (call at page end)."""
    st.markdown('---')
    st.markdown('#### Important information')
    st.caption(config.DISCLAIMER)
    st.caption('Market data may be delayed or temporarily unavailable. Historical '
               'model outputs are estimates, not forecasts or trading signals.')


def refresh_button(container=None) -> bool:
    holder = container if container is not None else st
    return holder.button('🔄 Refresh market data',
                         help='Re-download prices from Yahoo Finance and rerun the model.')


def constraint_check_pills(result) -> None:
    ok_all = all(all(v[k] for k in ('sum_to_one', 'no_shorts', 'cap_respected'))
                 for v in result['constraint_checks'].values())
    if ok_all:
        st.success('Constraint check passed: all portfolios sum to 100%, '
                   'no short selling, no asset above the 15% cap.')
    else:
        failed = {n: v for n, v in result['constraint_checks'].items()
                  if not all(v[k] for k in ('sum_to_one', 'no_shorts', 'cap_respected'))}
        st.error(f'Constraint check FAILED for: {list(failed)} — '
                 'official results are not displayed when constraints are violated.')


GLOSSARY = {
    'Expected Return': 'Historical annualized return estimate, not a promise.',
    'Volatility': 'Annualized variability of daily returns.',
    'Sharpe Ratio': 'Return above the risk-free rate per unit of volatility.',
    'Sortino Ratio': 'Risk-adjusted return that focuses on downside variation.',
    'Maximum Drawdown': 'Worst observed peak-to-trough loss in the selected period.',
    'CVaR / Expected Shortfall': 'Average loss among the worst 5% of observed days.',
    'VaR (Value at Risk)': 'Historical loss threshold exceeded on 5% of observed days.',
    'Calmar Ratio': 'Annualized return relative to maximum drawdown.',
    'Ledoit-Wolf Shrinkage': 'Stabilizes covariance estimates by blending with a structured target.',
    'K-Medoids Clustering': 'Groups assets using a correlation-derived distance.',
    'Walk-Forward Backtest': 'Repeatedly fits on past data and evaluates on unseen periods.',
    'HHI / Effective N': 'Concentration measure; effective N is the equivalent number of equal positions.',
    'Risk Contribution': 'Each asset’s share of portfolio variance, including co-movement.',
    'Out-of-Sample (OOS)': 'Evaluation data withheld from model fitting.',
}
