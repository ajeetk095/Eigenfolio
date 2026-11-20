"""Page 5 — Risk Analytics with plain-language explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.risk_metrics import (daily_var_es, drawdown_series, effective_n,
                              portfolio_metrics, risk_contributions)
from src.ui_utils import (data_freshness_bar, disclaimer_box, get_result,
                          metric_tooltip, refresh_button)
from src import visualizations as viz

st.set_page_config(page_title='Risk Analytics', page_icon='🛡️', layout='wide')
st.title('🛡️ Risk Analytics')

refresh = refresh_button()
try:
    result = get_result(force_refresh=refresh)
except Exception as exc:                       # noqa: BLE001
    st.error(f'Market data unavailable: {exc}')
    st.stop()

data_freshness_bar(result)
method = st.selectbox('Portfolio', list(result['weights'].keys()),
                      index=0, help='Choose which portfolio to analyse.')
w = result['weights'][method]
ret_frame = result['train_ret']
metrics = result['metrics'][method]['train']
S = result['S_sample']

# ------------------------------------------------------------ headline ------
st.subheader('Headline risk metrics — training period')
k1, k2, k3, k4 = st.columns(4)
k1.metric('Annualized volatility', f"{metrics['Ann. Vol']:.2%}")
k2.metric('Maximum drawdown', f"{metrics['Max Drawdown']:.2%}")
k3.metric('CVaR 95% (ann.)', f"{metrics['CVaR 95%']:.2%}")
k4.metric('Sharpe / Sortino',
          f"{metrics['Sharpe']:.2f} / {metrics['Sortino']:.2f}")

metric_tooltip(
    'Annualized Volatility', 'How much daily returns swing around, scaled to a year.',
    'It is the day-to-day bumpiness you must be able to tolerate.',
    f"{metrics['Ann. Vol']:.0%} implies daily swings around "
    f"{metrics['Ann. Vol'] / 252 ** 0.5:.1%} on average.",
    'Volatility is symmetric — it measures ups as well as downs, and it is '
    'estimated from history.')
metric_tooltip(
    'Maximum Drawdown', 'The worst loss from a peak to a trough in the period.',
    'It shows the deepest hole an investor would have experienced.',
    f"A value of {metrics['Max Drawdown']:.0%} means at the worst point the "
    'portfolio was that far below its previous high.',
    'Past drawdowns do not bound future ones.')
metric_tooltip(
    'CVaR / Expected Shortfall (95%)',
    'The average loss on the worst 5% of days.',
    'It describes a typical "very bad day", not just a threshold.',
    'Computed on daily returns; the report annualises by √252 — the same '
    'convention is used here for comparability.',
    'Based on only ~373 days; true tail events are rare and under-sampled.')
metric_tooltip(
    'Sharpe & Sortino ratios', 'Return per unit of risk (Sortino counts downside only).',
    'They let you compare portfolios on a risk-adjusted basis.',
    'Above ~1 is generally considered good.',
    'Sensitive to the risk-free assumption (5.25% p.a. here) and to the '
    'estimation window.')

# ------------------------------------------------------ risk contribution ---
st.subheader('Weight vs. risk contribution')
rc = risk_contributions(w, S)
st.plotly_chart(viz.risk_contribution_chart(rc, pd.Series(w, index=config.TICKERS)),
                width='stretch')
c1, c2, c3 = st.columns(3)
c1.metric('Effective positions (1/HHI)', f'{effective_n(w):.1f}')
c2.metric('Top risk contributor', f'{rc.idxmax()} ({rc.max() * 100:.0f}%)')
c3.metric('Assets at 15% cap',
          f"{int(sum(1 for x in w if x >= 0.15 - 1e-6))}")

with st.expander('🔍 Why can risk contribution differ from weight?'):
    st.markdown('An asset that moves together with the rest of the portfolio '
                'adds *more* risk than its weight suggests, because it falls '
                'exactly when everything else falls. Risk contribution counts '
                'both volatility and co-movement: RC_i = w_i (Σw)_i / (w᾿Σw).')

# --------------------------------------------------------------- VaR/ES -----
st.subheader('Daily VaR & Expected Shortfall (historical, 95%)')
port_daily = ret_frame.values @ np.asarray(w)
vares = daily_var_es(port_daily, alpha=0.05)
v1, v2 = st.columns(2)
v1.metric('Daily VaR (95%)', f'{vares["VaR"]:.2%}',
          help='On 5% of days, losses exceeded this level (daily, historical method).')
v2.metric('Daily ES / CVaR (95%)', f'{vares["ES"]:.2%}',
          help='Average loss across those worst 5% of days.')
st.caption('Convention: historical (non-parametric) daily log returns, '
           'non-annualised figures as displayed; annualised variants in the '
           'tables follow the report convention (√252).')

# ----------------------------------------------------------- sensitivity ----
st.subheader('Sensitivity — what if an asset is dropped?')
st.caption('Each row redistributes one asset\u2019s weight pro-rata across the '
           'others and recomputes the training metrics — a simple robustness '
           'check, not a forecast.')
sens_rows = []
tickers = config.TICKERS
for i, t in enumerate(tickers):
    if w[i] <= 1e-6:
        continue
    w2 = np.array(w, dtype=float)
    rest = [j for j in range(len(w)) if j != i]
    share = w2[rest] / w2[rest].sum()
    w2[rest] = share * (w2[i] + w2[rest].sum() - w2[rest].sum() + w2[rest].sum()) \
        if False else share * (w2[rest].sum() + w2[i])
    w2 /= w2.sum()
    m = portfolio_metrics(w2, ret_frame)
    sens_rows.append({'Dropped asset': t,
                      'Ann. return': m['Ann. Return'],
                      'Ann. vol': m['Ann. Vol'],
                      'Sharpe': m['Sharpe'],
                      'Max DD': m['Max Drawdown']})
sens = pd.DataFrame(sens_rows).sort_values('Sharpe')
st.dataframe(sens.style.format({
    'Ann. return': '{:.2%}', 'Ann. vol': '{:.2%}',
    'Sharpe': '{:.3f}', 'Max DD': '{:.2%}'}), width='stretch')
st.caption('Reading guide: a large Sharpe drop when removing an asset flags '
           'heavy dependence on that single name — a concentration signal.')

st.subheader('Asset volatility & correlation (training period)')
a1, a2 = st.columns([1, 2])
with a1:
    asset_vol = pd.DataFrame({'Ann. volatility': ret_frame.std() * 252 ** 0.5})
    st.dataframe(asset_vol.style.format('{:.1%}'))
with a2:
    st.plotly_chart(viz.correlation_heatmap(
        ret_frame.corr(), 'Correlation matrix (training)'),
        width='stretch')

disclaimer_box()
