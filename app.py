"""
QUANT PORTFOLIO INTELLIGENCE — Home / Investment Intelligence page.

Implements the GWP3 Streamlit application built directly on the original
MScFE 652 GWP3 model (baseline MVO, Ledoit-Wolf denoising, K-Medoids
clustering, combination portfolio, walk-forward backtest).

Run:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.pipeline import metrics_table, winners_per_metric
from src.risk_metrics import cumulative_returns
from src.recommendations import generate_recommendations, audience_view
from src.risk_metrics import effective_n, risk_contributions
from src.ui_utils import (constraint_check_pills, data_freshness_bar,
                          disclaimer_box, get_result, refresh_button,
                          weights_dict)
from src import visualizations as viz

st.set_page_config(
    page_title='Quant Portfolio Intelligence',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded',
)


def _fmt(x: float) -> str:
    """Signed percentage formatting for live quote notes."""
    return f'{x:+.2f}%'

# ---------------------------------------------------------------- header ----
st.markdown(
    """
    <div style="padding:1.2rem 1.5rem;border-radius:12px;
    background:linear-gradient(120deg,#0d1b2a 0%,#1b4965 60%,#2c7da0 100%);">
      <h1 style="color:#ffffff;margin:0;">📈 QUANT PORTFOLIO INTELLIGENCE</h1>
      <p style="color:#dbe9f4;margin:0.3rem 0 0 0;font-size:1.05rem;">
        Machine Learning &nbsp;|&nbsp; Quantitative Portfolio Optimization
        &nbsp;|&nbsp; Investment Risk Analytics</p>
      <p style="color:#a9cbe3;margin:0.2rem 0 0 0;font-size:0.95rem;">
        <b>Understand Markets. Optimize Portfolios. Manage Risk.</b></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('---')

# ------------------------------------------------------------ data + model --
refreshed = refresh_button()
try:
    result = get_result(force_refresh=refreshed)
except Exception as exc:                       # noqa: BLE001
    st.error(f'Market data could not be loaded: {exc}')
    st.info('This application computes every result from live Yahoo Finance '
            'data (via yfinance). When the data provider is unreachable, '
            'nothing is displayed rather than showing stale or invented '
            'numbers. Please press Refresh or try again later.')
    st.stop()

data_freshness_bar(result)
constraint_check_pills(result)

methods = list(result['weights'].keys())
train_tbl = metrics_table(result['metrics'], 'train')
test_tbl = metrics_table(result['metrics'], 'test')
base_w = result['weights']['Baseline (MVO)']

# ------------------------------------------------------- what is this app? --
col_info, col_how = st.columns([1.4, 1])
with col_info:
    st.subheader('What is this platform?')
    st.markdown(
        'An interactive quantitative portfolio intelligence platform built on a '
        'completed **MScFE 652 (WorldQuant University) Group Work Project 3** '
        'research model: a 12-asset U.S. large-cap portfolio, mean-variance '
        'optimization, and machine-learning improvements — **Ledoit-Wolf '
        'covariance denoising**, **K-Medoids clustering**, and a **rolling '
        'walk-forward backtest** — validated out-of-sample on July 2025.')
    st.markdown(
        '**What can you do here?**\n'
        '- Explore financial market data\n'
        '- Understand portfolio allocations\n'
        '- Compare traditional and ML-enhanced portfolios\n'
        '- Examine diversification and concentration\n'
        '- Analyse historical performance and risk\n'
        '- Explore research-driven investment insights')
with col_how:
    st.subheader('How to use it')
    st.markdown(
        '1. Start here for the headline view and today’s recommendations.\n'
        '2. **Live Market Dashboard** — prices, changes, correlations.\n'
        '3. **Portfolio Optimization** — allocations and the efficient frontier.\n'
        '4. **Denoising & Clustering** — the ML machinery, visualized.\n'
        '5. **Backtesting & Performance** — train vs. out-of-sample.\n'
        '6. **Risk Analytics** — VaR, ES, drawdown, risk contributions.\n'
        '7. **Investment Insights** — the full recommendations engine.\n'
        '8. **Methodology** — the academic project explained simply.')

# ----------------------------------------------------------- KPI headline ---
st.subheader('Portfolio overview & key indicators')
kpi = result['metrics']
live_quotes = None
try:
    from src.data_loader import fetch_latest_quotes
    live_quotes = fetch_latest_quotes(config.TICKERS)
except Exception:                              # noqa: BLE001
    pass

c1, c2, c3, c4 = st.columns(4)
c1.metric('Baseline Sharpe (training)',
          f"{kpi['Baseline (MVO)']['train']['Sharpe']:.3f}",
          help='Max-Sharpe MVO on Jan 2024–Jun 2025, long-only, 15% cap.')
c2.metric('Baseline Sharpe (OOS Jul 2025)',
          f"{kpi['Baseline (MVO)']['test']['Sharpe']:.3f}",
          f"{(kpi['Baseline (MVO)']['test']['Sharpe'] - kpi['Baseline (MVO)']['train']['Sharpe']):.3f} vs training",
          help='Required out-of-sample test: 21 trading days.')
c3.metric('Combo ann. vol (training)',
          f"{kpi['Combo (Den+Clust)']['train']['Ann. Vol']:.2%}",
          help='Lowest training volatility of all four portfolios.')
c4.metric('LW shrinkage α*',
          f"{result['denoise']['alpha']:.4f}",
          help='Share of covariance pulled toward the structured target.')

st.caption('These are the **freshly recomputed** model results from the current '
           'data download. The figures published in the submitted GWP3 report '
           '(computed Jan–Jul 2025) may differ slightly with newer data — that '
           'is expected and disclosed, never hidden.')

# ------------------------------------------------- risk assessment banner ---
eff_n_base = effective_n(base_w)
rc = risk_contributions(base_w, result['S_sample'])
risk_level = ('Elevated' if eff_n_base < 4 or
              kpi['Baseline (MVO)']['train']['Ann. Vol'] > 0.20 else 'Moderate')
st.markdown(f'### Current portfolio risk assessment: **{risk_level}**')
ra1, ra2, ra3, ra4 = st.columns(4)
ra1.metric('Effective positions', f'{eff_n_base:.1f}')
ra2.metric('Max asset weight', f'{max(base_w) * 100:.1f}% (cap 15%)')
ra3.metric('Top risk contributor',
           f'{rc.idxmax()} ({rc.max() * 100:.0f}%)')
ra4.metric('Training max drawdown',
           f"{kpi['Baseline (MVO)']['train']['Max Drawdown']:.1%}")

# ------------------------------------------------------ recommendations ------
st.markdown('### 🔑 Key recommendations (generated from actual model outputs)')
live_note = None
if live_quotes is not None and len(live_quotes['quotes']):
    q = live_quotes['quotes']
    worst = q.loc[q['Change %'].idxmin()]
    best = q.loc[q['Change %'].idxmax()]
    live_note = (f'Latest session: {best["Ticker"]} {_fmt(best["Change %"])} vs '
                 f'{worst["Ticker"]} {_fmt(worst["Change %"])} (delayed data).')
recs = generate_recommendations(result, live_vol_note=live_note)
show_recs = audience_view(recs, 'Investors')
for r in show_recs[:4]:
    with st.expander(f"**{r['title']}**"):
        st.markdown(f"**What the analysis found:** {r['finding']}")
        st.markdown(f"**Why it matters:** {r['why']}")
        st.markdown(f"**Assets involved:** {r['assets']}")
        st.markdown(f"**What to consider:** {r['consider']}")
        st.markdown(f"**Risks / uncertainties:** {r['risks']}")
        st.caption(f"Supporting metrics: {r['supporting']}")
st.info('The full recommendations engine — with separate views for investors, '
        'retail investors and companies — is on the **Investment Insights** page.')


# ------------------------------------------------- audience insight cards ---
st.markdown('### Three ways to read these results')
a1, a2, a3 = st.columns(3)
with a1:
    st.markdown('#### 👤 For investors')
    st.markdown(
        f"The baseline portfolio earned {kpi['Baseline (MVO)']['train']['Ann. Return']:.1%} "
        f"annualised in training with {kpi['Baseline (MVO)']['train']['Ann. Vol']:.1%} "
        f"volatility, but only {kpi['Baseline (MVO)']['test']['Ann. Return']:.1%} "
        'annualised out-of-sample — a cautionary illustration of overfitting. '
        'Allocation, risk and diversification details are on every page.')
with a2:
    st.markdown('#### 🛒 For retail investors')
    st.markdown(
        'In plain terms: the model picks a mix of 12 well-known U.S. stocks '
        '(Apple, Microsoft, Nvidia, Costco…) with two hard rules — never bet '
        'against a stock, and never put more than 15% of money in one. The '
        'ML tools try to make that mix more stable. Every metric has a '
        '“What does this mean?” explainer — no finance degree required.')
with a3:
    st.markdown('#### 🏢 For companies & financial institutions')
    st.markdown(
        'The analysis quantifies asset exposure and concentration (HHI, '
        f'effective N = {eff_n_base:.1f}), risk contributions, and regime '
        'sensitivity: 9 of 12 assets cluster into one technology factor, so '
        'diversification within this universe is limited. Cluster-level '
        'allocation and covariance conditioning diagnostics are provided.')

# ------------------------------------------------------------- key visuals --
st.markdown('### Market & portfolio snapshot')
try:
    prices_recent = result['prices'].tail(252)
    st.plotly_chart(viz.normalized_price_chart(
        prices_recent, config.TICKERS), use_container_width=True)
except Exception:                              # noqa: BLE001
    pass

st.plotly_chart(viz.cumulative_return_chart(
    {name: cumulative_returns(w, result['test_ret'])
     for name, w in result['weights'].items()},
    'Out-of-Sample Cumulative Returns — July 2025 (required GWP3 test)'),
    use_container_width=True)

# ----------------------------------------------------------- market pulse ---
st.markdown('### Market conditions summary')
if live_quotes is not None and len(live_quotes['quotes']):
    q = live_quotes['quotes'].copy()
    adv = int((q['Change %'] > 0).sum())
    st.markdown(
        f"Latest session breadth: **{adv} of {len(q)}** assets closed higher "
        f"(delayed end-of-day data, retrieved {live_quotes['fetched_at']} UTC). "
        'See the Live Market Dashboard for the full picture.')
else:
    st.caption('Live quotes are temporarily unavailable — historical analysis '
               'remains fully available.')

disclaimer_box()
st.markdown('---')
st.caption('Built on the MScFE 652 GWP3 research model · '
           '[GitHub](https://github.com/ajeetk095) · '
           '[LinkedIn](https://www.linkedin.com/in/ajeek095/) · '
           '[About the developer](/8_About_Developer)')
