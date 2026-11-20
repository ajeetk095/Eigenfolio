"""Page 2 — Portfolio Optimization Lab."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.pipeline import metrics_table
from src.ui_utils import (constraint_check_pills, data_freshness_bar,
                          disclaimer_box, get_result, refresh_button,
                          weights_dict)
from src import visualizations as viz

st.set_page_config(page_title='Portfolio Optimization', page_icon='⚖️', layout='wide')
st.title('⚖️ Portfolio Optimization Lab')
st.caption('Official GWP3 portfolios are computed with the assignment’s hard '
           'rules: long-only, every asset ≤ 15%, fully invested.')

refresh = refresh_button()
try:
    result = get_result(force_refresh=refresh)
except Exception as exc:                       # noqa: BLE001
    st.error(f'Market data unavailable: {exc}')
    st.stop()

data_freshness_bar(result)
constraint_check_pills(result)

methods = list(result['weights'].keys())
labels = {
    'Baseline (MVO)': ('Baseline (MVO)', 'Step 1 — classic mean-variance max-Sharpe '
                       'with sample covariance'),
    'Denoise (LW)': ('Denoise (LW)', 'Step 2/3 — Ledoit-Wolf shrunk covariance, '
                     'same constraints'),
    'Cluster (K-Med)': ('Cluster (K-Med)', 'Step 2/3 — K-Medoids clusters with '
                       'inverse-volatility risk parity'),
    'Combo (Den+Clust)': ('Combo (Den+Clust)', 'Step 3 — Denoise → Cluster → MVO '
                         'combination pipeline'),
}

tab_names = [labels[m][0] for m in methods]
tabs = st.tabs(tab_names)
for tab, m in zip(tabs, methods):
    with tab:
        st.markdown(f'**{labels[m][1]}**')
        wd = weights_dict(result, m)
        df = pd.DataFrame({
            'Ticker': list(wd), 'Weight': list(wd.values()),
            'Sector': [config.SECTORS[t] for t in wd],
            'At 15% cap?': ['Yes' if w >= 0.15 - 1e-6 else 'No' for w in wd.values()],
            'Description': [config.ASSET_DESCRIPTIONS[t] for t in wd],
        })
        df = df.sort_values('Weight', ascending=False).reset_index(drop=True)
        df['Weight'] = df['Weight'].map('{:.2%}'.format)
        st.dataframe(df, width='stretch', hide_index=True)
        check = result['constraint_checks'][m]
        st.caption(f"Sum of weights = {check['sum']:.6f} · max weight = "
                   f"{check['max_weight'] * 100:.2f}% · shorts = "
                   f"{'none' if check['no_shorts'] else 'FOUND (invalid)'}")

        mt = metrics_table(result['metrics'], 'train').loc[m]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric('Expected return (train)', f"{mt['Ann. Return']:.2%}")
        k2.metric('Volatility (train)', f"{mt['Ann. Vol']:.2%}")
        k3.metric('Sharpe', f"{mt['Sharpe']:.3f}")
        k4.metric('Sortino', f"{mt['Sortino']:.3f}")
        st.plotly_chart(viz.allocation_pie(
            wd, f'{labels[m][0]} — allocation'), width='stretch')

st.markdown('---')
st.subheader('Risk–return map & efficient frontier')
st.plotly_chart(viz.frontier_chart(result['frontier'], result['weights'],
                                   result['mu'], result['S_sample']),
                width='stretch')
st.caption('Stars mark each portfolio; grey dots are individual assets. '
           'Estimates come from the training window and are not forecasts.')

st.subheader('Weights across all methods')
st.plotly_chart(viz.weights_bar_chart(
    {m: list(result['weights'][m]) for m in methods}, config.TICKERS),
    width='stretch')

st.subheader('Optimization settings (as documented in the GWP3 report)')
st.markdown(
    """
| Setting | Value |
|---|---|
| Objective | Maximize Sharpe ratio: max (w'μ − rf) / √(w'Σw) |
| Constraints | Σw = 1 · 0 ≤ w ≤ 0.15 · no short selling |
| Expected returns μ | Annualised mean of daily log returns (training window) |
| Covariance Σ | Sample (baseline) or Ledoit-Wolf shrunk (denoised), ×252 |
| Risk-free rate | 5.25% p.a. |
| Annualisation | 252 trading days (disclosed convention) |
| Transaction costs | Not modelled — single-shot static weights (documented limitation) |
""")
st.caption('Any user-defined scenario with different constraints would be a '
           'research extension and is deliberately NOT mixed with the official '
           'GWP3 results above.')

with st.expander('📋 Disclosed discrepancy vs. the submitted report (weight capping)'):
    st.markdown("""The original notebook enforced the 15% cap with a single
clip-then-renormalise pass; after renormalisation a weight can land **back above**
the cap — which is why the submitted report's Table 10 shows COST at **24.14%**
in the final portfolio. This app enforces the cap correctly with iterative
capping, so its freshly computed Combo weights differ from the report's Table 10
on exactly this point. The academic submission is preserved as published; the
live app applies the assignment rule as written (no asset above 15%). Both
versions are clearly labelled and never mixed.""")

disclaimer_box()
