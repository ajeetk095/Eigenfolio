"""Page 6 — Investment Insights & Recommendations dashboard."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, visualizations as viz
from src.pipeline import metrics_table, winners_per_metric
from src.recommendations import audience_view, generate_recommendations
from src.risk_metrics import cumulative_returns, risk_contributions
from src.ui_utils import (GLOSSARY, data_freshness_bar, disclaimer_box,
                          get_result, refresh_button)

st.set_page_config(page_title='Investment Insights', page_icon='💡', layout='wide')
st.title('💡 Investment Insights & Recommendations')
st.caption('Explore computed portfolio performance, allocations, risk signals, and audience-specific recommendations. These are historical model results—not forecasts or trading signals.')

refresh = refresh_button()
try:
    result = get_result(force_refresh=refresh)
except Exception as exc:  # noqa: BLE001
    st.error(f'Market data unavailable: {exc}')
    st.info('No charts or metrics are shown unless the model has valid market data. Please retry the sync when the provider is available.')
    st.stop()

data_freshness_bar(result)
st.info('A sync refreshes the source data used by the model, but the academic comparisons below retain their fixed training and out-of-sample windows. The trend charts show historical results, not a forecast.')

metrics = result['metrics']
weights = result['weights']
train_tbl = metrics_table(metrics, 'train')
test_tbl = metrics_table(metrics, 'test')
train_winners = winners_per_metric(train_tbl)
test_winners = winners_per_metric(test_tbl)

# A quick, evidence-backed snapshot keeps the key computed outputs visible.
sharpe_winner = train_winners['Sharpe']
return_winner = train_winners['Ann. Return']
oos_return_winner = test_winners['Ann. Return']
k1, k2, k3, k4 = st.columns(4)
k1.metric('Top training Sharpe', sharpe_winner,
          f"{train_tbl.loc[sharpe_winner, 'Sharpe']:.2f}")
k2.metric('Top training return', return_winner,
          f"{train_tbl.loc[return_winner, 'Ann. Return']:.1%} annualised")
k3.metric('Top OOS return', oos_return_winner,
          f"{test_tbl.loc[oos_return_winner, 'Ann. Return']:.1%} annualised")
k4.metric('Ledoit–Wolf shrinkage', f"{result['denoise']['alpha']:.3f}",
          'Training-period estimate')

# Performance trends and the underlying comparison table.
st.markdown('## 📈 Performance & trends')
period_choice = st.radio('Historical period',
                         ['Training period', 'Out-of-sample test'],
                         horizontal=True, key='insights_period')
if period_choice == 'Training period':
    period_returns = result['train_ret']
    period_label = 'Training period (Jan 2024 – Jun 2025)'
    metrics_table_for_period = train_tbl
    period_winners = train_winners
else:
    period_returns = result['test_ret']
    period_label = 'Out-of-sample test (July 2025)'
    metrics_table_for_period = test_tbl
    period_winners = test_winners

if period_returns.empty:
    st.warning(f'No return observations are available for the {period_choice.lower()}.')
else:
    performance = {
        name: cumulative_returns(weight, period_returns)
        for name, weight in weights.items()
    }
    st.plotly_chart(viz.cumulative_return_chart(
        performance, f'Cumulative historical return — {period_label}'),
        width='stretch', key='insights_performance_trend')
    st.caption(f'{len(period_returns)} trading-day observations. Historical performance can differ materially from future results.')

    formatted_metrics = metrics_table_for_period.style.format({
        'Ann. Return': '{:.2%}', 'Ann. Vol': '{:.2%}', 'Sharpe': '{:.3f}',
        'Sortino': '{:.3f}', 'Max Drawdown': '{:.2%}', 'CVaR 95%': '{:.2%}',
        'Calmar': '{:.3f}',
    })
    st.markdown('### Model comparison')
    st.dataframe(formatted_metrics, width='stretch')
    winner_frame = pd.DataFrame.from_dict(period_winners, orient='index',
                                          columns=['Best method'])
    st.caption('Best method by metric for this period (lower annualised volatility is better; higher is better for the other displayed metrics).')
    st.dataframe(winner_frame, width='stretch')

# Portfolio allocation visualizations and exact weights.
st.markdown('## 🧩 Portfolio mix')
allocation_methods = list(weights)
left, right = st.columns([1.2, 1])
with left:
    st.plotly_chart(viz.weights_bar_chart(weights, list(result['mu'].index)),
                    width='stretch', key='insights_weights')
with right:
    selected_method = st.selectbox('Allocation detail', allocation_methods,
                                   key='insights_allocation_method')
    allocation = pd.DataFrame({
        'Asset': list(result['mu'].index),
        'Weight': weights[selected_method],
    }).sort_values('Weight', ascending=False)
    st.dataframe(allocation.style.format({'Weight': '{:.2%}'}),
                 width='stretch', hide_index=True)
    st.caption('Weights are model estimates from the stated training window; the dashed chart line marks the assignment’s 15% cap.')

# Risk comparison is derived from the training covariance and baseline portfolio.
st.markdown('## 🛡️ Risk signals')
base_rc = risk_contributions(weights['Baseline (MVO)'], result['S_sample'])
base_weights = pd.Series(weights['Baseline (MVO)'], index=result['mu'].index)
risk_left, risk_right = st.columns([1.2, 1])
with risk_left:
    st.plotly_chart(viz.risk_contribution_chart(base_rc, base_weights),
                    width='stretch', key='insights_risk_contributions')
with risk_right:
    risk_table = pd.DataFrame({
        'Asset weight': base_weights,
        'Share of portfolio variance': base_rc,
    }).sort_values('Share of portfolio variance', ascending=False)
    st.dataframe(risk_table.style.format('{:.2%}'), width='stretch')
    st.caption('Risk contributions are estimated from training-period covariance; they can change as volatility and correlations change.')

# Recommendations remain visible even if a particular audience has no matches.
st.markdown('## ✨ Recommendations & research observations')
audience = st.radio(
    'Choose your view',
    ['For investors', 'For retail investors',
     'For companies & financial institutions'],
    horizontal=True,
)
recs = generate_recommendations(result)
audience_key = {
    'For investors': 'Investors',
    'For retail investors': 'Retail investors',
    'For companies & financial institutions': 'Companies',
}[audience]
view = audience_view(recs, audience_key)
if not view:
    st.info('No audience-specific recommendations are available for this view yet. The computed charts and tables above remain available.')
else:
    st.caption(f'{len(view)} evidence-backed observations for this audience, generated from the current model outputs.')
    for i, recommendation in enumerate(view):
        with st.expander(f"{i + 1}. {recommendation['title']}", expanded=(i == 0)):
            finding_col, context_col = st.columns(2)
            with finding_col:
                st.markdown('**What the analysis found**')
                st.write(recommendation['finding'])
                st.markdown('**Why it matters**')
                st.write(recommendation['why'])
                st.markdown('**Assets / characteristics involved**')
                st.write(recommendation['assets'])
            with context_col:
                st.markdown('**What to consider**')
                st.write(recommendation['consider'])
                st.markdown('**Risks & uncertainties**')
                st.write(recommendation['risks'])
                st.markdown('**Supporting metrics**')
                st.write(recommendation['supporting'])
            st.caption(f"Audience focus: {recommendation['audience']}")

with st.expander('How to read these recommendations and trends'):
    st.markdown('''
    - Recommendations describe estimated historical model behaviour; they are research observations, not trading signals.
    - Training results and the July 2025 out-of-sample test are shown separately. Rankings can change between periods.
    - Cumulative-return charts describe past model performance. They do not predict future returns.
    - Market data may be delayed or temporarily unavailable; a successful sync is not a guarantee of real-time quotes.
    ''')

with st.expander('Simple glossary'):
    for term, definition in GLOSSARY.items():
        st.markdown(f'- **{term}** — {definition}')

disclaimer_box()
