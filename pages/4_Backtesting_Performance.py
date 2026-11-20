"""Page 4 — Backtesting & Performance (train vs. required OOS test)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.pipeline import metrics_table, winners_per_metric
from src.risk_metrics import cumulative_returns, drawdown_series
from src.ui_utils import data_freshness_bar, disclaimer_box, get_result, refresh_button
from src import visualizations as viz

st.set_page_config(page_title='Backtesting & Performance', page_icon='📊',
                   layout='wide')
st.title('📊 Backtesting & Performance')

refresh = refresh_button()
try:
    result = get_result(force_refresh=refresh)
except Exception as exc:                       # noqa: BLE001
    st.error(f'Market data unavailable: {exc}')
    st.stop()

data_freshness_bar(result)
train_tbl = metrics_table(result['metrics'], 'train')
test_tbl = metrics_table(result['metrics'], 'test')
winners_train = winners_per_metric(train_tbl)
winners_test = winners_per_metric(test_tbl)

st.markdown('**Two separated evaluation environments:** the academic training '
            'window (Jan 2024 – Jun 2025), the assignment-required '
            'out-of-sample test (July 2025), plus an additional rolling '
            'walk-forward backtest. Nothing is mixed, nothing hidden — '
            'negative results are shown as they are.')

# ------------------------------------------------------------- tables -------
def styled_table(tbl: pd.DataFrame, winners: dict):
    fmt = {c: ('{:.2%}' if c in ('Ann. Return', 'Ann. Vol', 'Max Drawdown',
                                 'CVaR 95%') else '{:.3f}')
           for c in tbl.columns}
    out = tbl.copy()
    for c in tbl.columns:
        out[c] = out[c].map(fmt[c].format)
        best = winners[c]
        if best in out.index:
            out.loc[best, c] = f'**{out.loc[best, c]}** 🏆'
    return out


t1, t2 = st.tabs(['🏫 Training period (Jan 2024 – Jun 2025)',
                  '🧪 Out-of-sample test (July 2025)'])
with t1:
    st.dataframe(styled_table(train_tbl, winners_train), width='stretch')
    st.caption('Greenfield estimates: fitted on the same window, so these '
               'numbers are optimistic by construction.')
with t2:
    st.dataframe(styled_table(test_tbl, winners_test), width='stretch')
    st.caption('Required GWP3 test: 21 trading days the model never saw. '
               'The combination portfolio underperformed on return metrics '
               'here in the submitted report — that honest negative result '
               'is preserved.')

st.markdown('### Winner per metric')
w1, w2 = st.columns(2)
with w1:
    st.markdown('**Training**')
    st.dataframe(pd.DataFrame({'Best method': winners_train}), width='stretch')
with w2:
    st.markdown('**Out-of-sample (July 2025)**')
    st.dataframe(pd.DataFrame({'Best method': winners_test}), width='stretch')

# ------------------------------------------------------------- charts -------
st.markdown('### Cumulative performance')
period = st.radio('Period', ['Training (Jan 2024 – Jun 2025)', 'OOS test (July 2025)'],
                  horizontal=True)
ret_frame = result['train_ret'] if period.startswith('Training') else result['test_ret']
cum_map = {name: cumulative_returns(w, ret_frame)
           for name, w in result['weights'].items()}
st.plotly_chart(viz.cumulative_return_chart(
    cum_map, f'Cumulative Returns — {period}'), width='stretch')

st.plotly_chart(viz.drawdown_chart(
    {name: drawdown_series(w, ret_frame) for name, w in result['weights'].items()}),
    width='stretch')

st.markdown('### Rolling volatility (60-day window, training period)')
roll_vol = pd.DataFrame({
    name: (ret_frame @ w).rolling(60).std() * 252 ** 0.5
    for name, w in result['weights'].items()}).dropna()
st.line_chart(roll_vol)

# ------------------------------------------------------- walk-forward -------
st.markdown('### Additional analysis — rolling walk-forward backtest')
st.caption('Clearly an **additional research component**: every month, the '
           'model re-fits on the previous ~6 months only (no look-ahead) and '
           'is judged on the next unseen month. This is distinct from the '
           'original GWP3 validation above.')
bt = result['backtest']
if bt is None or bt['windows'] is None or not len(bt['windows']):
    st.info('Walk-forward backtest was not run in this session.')
else:
    wdw = bt['windows']
    b1, b2, b3 = st.columns(3)
    b1.metric('Windows completed', len(wdw))
    b2.metric('Mean OOS Sharpe', f"{wdw['Sharpe'].mean():.3f}",
              f"± {wdw['Sharpe'].std():.3f}")
    b3.metric('Mean OOS ann. return', f"{wdw['Ann. Return'].mean():.2%}",
              f"± {wdw['Ann. Return'].std():.2%}")
    st.plotly_chart(viz.rolling_sharpe_chart(wdw), width='stretch')
    st.plotly_chart(viz.backtest_wealth_chart(bt['cum_oos']), width='stretch')
    st.dataframe(wdw.round(4), width='stretch')

st.markdown('### Why combinations can underperform')
st.markdown("""
The report's honest finding: in July 2025 the combination portfolio had the
best tail-risk profile (lowest CVaR, shallowest drawdown) but the *worst*
return metrics, because the month rewarded exactly the concentrated momentum
exposure (NVDA, META, NFLX) that clustering deliberately dilutes — and COST's
24% allocation lagged. A 21-day window captures one market regime; it cannot
validate or refute a structural advantage. The walk-forward analysis above
exists precisely to add more regimes to the evidence.
""")
disclaimer_box()
