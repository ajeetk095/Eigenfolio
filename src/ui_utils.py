"""
Shared Streamlit utilities: model bootstrap, UI helpers, glossary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make src/ importable when Streamlit runs app.py or pages/ scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.pipeline import build_full_result, metrics_table, winners_per_metric  # noqa: E402


# ---------------------------------------------------------------- bootstrap --
@st.cache_resource(show_spinner='Running the full GWP3 model on live market data…')
def get_result(force_refresh: bool = False):
    """Compute (and cache) the complete GWP3 result from current market data."""
    return build_full_result(force_refresh=force_refresh)


def weights_dict(result, method: str) -> dict:
    """Ticker -> weight mapping for display."""
    return {t: float(w) for t, w in zip(config.TICKERS, result['weights'][method])}


# -------------------------------------------------------------------- ui ----
def metric_tooltip(name: str, definition: str, why: str, how: str,
                   limit: str) -> None:
    st.markdown(f'**{name}**')
    with st.expander('What does this mean?'):
        st.markdown(
            f'- **Definition:** {definition}\n'
            f'- **Why it matters:** {why}\n'
            f'- **How to read it:** {how}\n'
            f'- **Limitations:** {limit}')


def data_freshness_bar(result) -> None:
    """Show data provenance for model or history payloads without brittle key errors.

    Model results use `data_source` and `validation`; raw loader payloads use
    `source`. Accept both schemas so the dashboard remains robust across callers.
    """
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
    st.info(config.DISCLAIMER)


def refresh_button(container=None) -> bool:
    holder = container if container is not None else st
    return holder.button('🔄 Refresh market data',
                         help='Re-download prices from Yahoo Finance and '
                              're-run the full model (cached for ~1 hour).')


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
                 'please report this; official results are not displayed '
                 'when constraints are violated.')


GLOSSARY = {
    'Expected Return': 'The average yearly return the portfolio would have earned '
                       'in the training period, based on history. An estimate, not a promise.',
    'Volatility': 'How much daily returns swing around. Annualised. Higher = bumpier ride.',
    'Sharpe Ratio': 'Return earned per unit of risk (above the risk-free rate). Above ~1 is '
                    'considered good; above ~2 is excellent. Does not protect against rare crashes.',
    'Sortino Ratio': 'Like Sharpe, but only penalises downside swings. Higher is better.',
    'Maximum Drawdown': 'The worst peak-to-trough loss in the period. -25% means you were '
                        'once down 25% from a high point.',
    'CVaR / Expected Shortfall': 'The average loss on the worst 5% of days. Shows what a '
                                 '"bad day" typically looks like in the tail.',
    'VaR (Value at Risk)': 'The loss threshold exceeded on only 5% of days.',
    'Calmar Ratio': 'Annualised return divided by the worst drawdown. Higher = more return '
                    'per unit of worst-case pain.',
    'Ledoit-Wolf Shrinkage': 'A statistical adjustment that pulls extreme estimates of how '
                             'assets co-move toward the average, making them more stable.',
    'K-Medoids Clustering': 'An ML method that groups assets that move together, so the '
                            'portfolio is not over-concentrated in one "team" of stocks.',
    'Walk-Forward Backtest': 'Re-optimising the portfolio every month using only past data, '
                             'then judging it on the next unseen month — no look-ahead cheating.',
    'HHI / Effective N': 'A concentration measure. Effective N = how many equally-sized '
                         'positions would give the same concentration.',
    'Risk Contribution': 'The share of total portfolio risk each asset is responsible for, '
                         'counting both its volatility and its co-movement.',
    'Out-of-Sample (OOS)': 'A test period the model never saw while fitting — the July 2025 '
                           'window required by the assignment.',
}
