"""Page 6 — date-range-aware investment insights dashboard."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src import config, data_loader, data_validation, visualizations as viz
from src.pipeline import metrics_table, winners_per_metric
from src.recommendations import audience_view, generate_recommendations
from src.risk_metrics import (cumulative_returns, portfolio_metrics,
                              risk_contributions)
from src.ui_utils import (GLOSSARY, data_freshness_bar, disclaimer_box,
                          get_result, refresh_button)

HISTORY_START = '2024-01-03'

st.set_page_config(page_title='Investment Insights', page_icon='💡', layout='wide')
st.title('💡 Investment Insights & Recommendations')
st.caption(
    'Explore portfolio performance and evidence-based observations for a '
    'visitor-selected historical range. Data is the latest available adjusted '
    'daily market data—not guaranteed intraday or tick-by-tick real time.'
)

refresh = refresh_button()
try:
    result = get_result(force_refresh=refresh)
    # The academic model keeps its fixed train/test windows. Fetch a separate,
    # up-to-date history for date-range views and current-period observations.
    history = data_loader.fetch_price_history(
        config.TICKERS, HISTORY_START, date.today().isoformat(),
        force_refresh=refresh,
    )
    prices = data_validation.clean_prices(history['prices'], config.TICKERS)
    history_returns = data_validation.compute_returns(prices)['log']
except Exception as exc:  # noqa: BLE001
    st.error(f'Market data unavailable: {exc}')
    st.info('No charts or metrics are shown unless the model has valid market data. Please retry the sync when the provider is available.')
    st.stop()

validation = data_validation.validate_data(prices, history_returns)
data_freshness_bar({**history, 'validation': validation})
if not validation['ok']:
    st.warning('The latest history did not pass all data-integrity checks; inspect results cautiously.')

if prices.empty or history_returns.empty:
    st.warning('No usable daily return observations are available for the selected history.')
    st.stop()

first_available = prices.index.min().date()
last_available = prices.index.max().date()
default_start = max(date.fromisoformat(HISTORY_START), first_available)

st.markdown('## 📅 Choose the market-history range')
st.caption('The end date is limited to the latest market observation available from the data provider. On market holidays or before the close, this can be earlier than today.')
selected_range = st.date_input(
    'Start and end dates',
    value=(default_start, last_available),
    min_value=first_available,
    max_value=last_available,
    key='insights_date_range',
)
if isinstance(selected_range, (tuple, list)) and len(selected_range) == 2:
    start_date, end_date = selected_range
elif isinstance(selected_range, (tuple, list)) and len(selected_range) == 1:
    start_date = end_date = selected_range[0]
else:
    start_date = end_date = selected_range

selected_returns = history_returns.loc[
    (history_returns.index.date >= start_date)
    & (history_returns.index.date <= end_date)
]
if selected_returns.empty:
    st.warning('There are no trading-day returns in that date range. Choose a wider range.')
    st.stop()
period_label = f'{start_date:%Y-%m-%d} → {end_date:%Y-%m-%d}'
st.caption(f'Selected period: **{period_label}** · {len(selected_returns)} trading-day return observations · Latest observation: {last_available:%Y-%m-%d}')

weights = result['weights']
metrics = result['metrics']
train_tbl = metrics_table(metrics, 'train')
test_tbl = metrics_table(metrics, 'test')
train_winners = winners_per_metric(train_tbl)
test_winners = winners_per_metric(test_tbl)

# Compute all comparisons on the same fixed model weights, using only returns
# inside the visitor-selected interval. This is historical analysis, not a live forecast.
range_metrics = pd.DataFrame({
    method: portfolio_metrics(weight, selected_returns)
    for method, weight in weights.items()
}).T
range_metrics.index.name = 'Method'
range_winners = winners_per_metric(range_metrics)

st.markdown('## 📈 Performance & trends')
trend_col, snapshot_col = st.columns([1.5, 1])
with trend_col:
    performance = {
        method: cumulative_returns(weight, selected_returns)
        for method, weight in weights.items()
    }
    st.plotly_chart(
        viz.cumulative_return_chart(
            performance, f'Cumulative historical return — {period_label}'),
        width='stretch', key='insights_performance_trend',
    )
with snapshot_col:
    st.markdown('### Selected-range snapshot')
    best_return = range_winners['Ann. Return']
    best_sharpe = range_winners['Sharpe']
    lowest_vol = range_winners['Ann. Vol']
    st.metric('Highest annualised return', best_return,
              f"{range_metrics.loc[best_return, 'Ann. Return']:.2%}")
    st.metric('Highest Sharpe ratio', best_sharpe,
              f"{range_metrics.loc[best_sharpe, 'Sharpe']:.3f}")
    st.metric('Lowest annualised volatility', lowest_vol,
              f"{range_metrics.loc[lowest_vol, 'Ann. Vol']:.2%}")
    st.caption('Annualised historical estimates based on this range; rankings are sensitive to the chosen dates.')

st.markdown('### Model comparison — selected range vs. academic reference')
selected_col, academic_col = st.columns(2)
with selected_col:
    st.markdown(f'**Visitor-selected range ({period_label})**')
    st.dataframe(range_metrics.style.format({
        'Ann. Return': '{:.2%}', 'Ann. Vol': '{:.2%}', 'Sharpe': '{:.3f}',
        'Sortino': '{:.3f}', 'Max Drawdown': '{:.2%}', 'CVaR 95%': '{:.2%}',
        'Calmar': '{:.3f}',
    }), width='stretch')
    st.caption('Best selected-range method by metric: ' + ', '.join(
        f'{metric}: {method}' for metric, method in range_winners.items()))
with academic_col:
    st.markdown('**Fixed academic reference windows**')
    st.caption('Training: Jan 2024–Jun 2025')
    st.dataframe(train_tbl.style.format({
        'Ann. Return': '{:.2%}', 'Ann. Vol': '{:.2%}', 'Sharpe': '{:.3f}',
        'Sortino': '{:.3f}', 'Max Drawdown': '{:.2%}', 'CVaR 95%': '{:.2%}',
        'Calmar': '{:.3f}',
    }), width='stretch')
    st.caption('Out-of-sample reference: July 2025')
    st.dataframe(test_tbl.style.format({
        'Ann. Return': '{:.2%}', 'Ann. Vol': '{:.2%}', 'Sharpe': '{:.3f}',
        'Sortino': '{:.3f}', 'Max Drawdown': '{:.2%}', 'CVaR 95%': '{:.2%}',
        'Calmar': '{:.3f}',
    }), width='stretch')

st.markdown('## 🧩 Portfolio mix')
allocation_left, allocation_right = st.columns([1.2, 1])
with allocation_left:
    st.plotly_chart(viz.weights_bar_chart(weights, list(result['mu'].index)),
                    width='stretch', key='insights_weights')
with allocation_right:
    selected_method = st.selectbox('Allocation detail', list(weights),
                                   key='insights_allocation_method')
    allocation = pd.DataFrame({
        'Asset': list(result['mu'].index),
        'Weight': weights[selected_method],
    }).sort_values('Weight', ascending=False)
    st.dataframe(allocation.style.format({'Weight': '{:.2%}'}),
                 width='stretch', hide_index=True)
    st.caption('These model weights are fitted on the fixed training window and are held constant when comparing the selected date range. They are not automatically re-optimised for each range.')

st.markdown('## 🛡️ Risk signals')
base_weights = pd.Series(weights['Baseline (MVO)'], index=result['mu'].index)
risk_range_col, risk_train_col = st.columns(2)
if len(selected_returns) >= 2:
    selected_cov = selected_returns.cov() * config.TRADING_DAYS
    selected_rc = risk_contributions(weights['Baseline (MVO)'], selected_cov)
    with risk_range_col:
        st.markdown(f'**Selected range ({period_label})**')
        st.plotly_chart(viz.risk_contribution_chart(selected_rc, base_weights),
                        width='stretch', key='insights_range_risk_contributions')
        selected_risk = pd.DataFrame({
            'Asset weight': base_weights,
            'Share of selected-range variance': selected_rc,
        }).sort_values('Share of selected-range variance', ascending=False)
        st.dataframe(selected_risk.style.format('{:.2%}'), width='stretch')
        st.caption('Covariance estimated from the selected period; short ranges can make risk estimates unstable.')
else:
    with risk_range_col:
        st.info('Select at least two daily return observations to estimate selected-range risk contributions.')

base_rc = risk_contributions(weights['Baseline (MVO)'], result['S_sample'])
with risk_train_col:
    st.markdown('**Fixed training-period reference**')
    st.plotly_chart(viz.risk_contribution_chart(base_rc, base_weights),
                    width='stretch', key='insights_training_risk_contributions')
    training_risk = pd.DataFrame({
        'Asset weight': base_weights,
        'Share of training-period variance': base_rc,
    }).sort_values('Share of training-period variance', ascending=False)
    st.dataframe(training_risk.style.format('{:.2%}'), width='stretch')
    st.caption('Training-period covariance; risk contributions change as volatility and correlations change.')

# Date-conditioned research observations are computed from the selected returns.
st.markdown('## ✨ Selected-range recommendations & observations')
st.caption('These are data-driven historical observations for the selected dates, not personalised financial advice or buy/sell signals.')
observation_col, risk_observation_col = st.columns(2)
with observation_col:
    st.markdown('### Performance observations')
    st.write(
        f'**{best_return}** had the highest annualised return in this range '
        f'({range_metrics.loc[best_return, "Ann. Return"]:.2%}); **{best_sharpe}** '
        f'had the highest Sharpe ratio '
        f'({range_metrics.loc[best_sharpe, "Sharpe"]:.3f}).'
    )
    st.write(
        f'**{lowest_vol}** had the lowest annualised volatility '
        f'({range_metrics.loc[lowest_vol, "Ann. Vol"]:.2%}). Consider comparing '
        'return, drawdown, and volatility together rather than relying on one ranking.'
    )
with risk_observation_col:
    st.markdown('### Risk observations')
    if len(selected_returns) >= 2:
        leading_risk_asset = selected_rc.idxmax()
        st.write(
            f'For the fixed Baseline (MVO) weights, **{leading_risk_asset}** was '
            f'the largest estimated contributor to portfolio variance in this range '
            f'({selected_rc.loc[leading_risk_asset]:.1%}).'
        )
    else:
        st.write('More observations are needed to calculate a selected-range risk contribution comparison.')
    st.write(
        f'The largest observed baseline drawdown among the compared portfolios was '
        f'{range_metrics["Max Drawdown"].min():.2%}. Historical risk estimates '
        'do not describe or guarantee future market behaviour.'
    )

with st.expander('Additional model research notes (fixed academic windows)'):
    audience = st.radio(
        'Choose an audience view',
        ['For investors', 'For retail investors',
         'For companies & financial institutions'],
        horizontal=True, key='insights_audience',
    )
    recs = generate_recommendations(result)
    audience_key = {
        'For investors': 'Investors',
        'For retail investors': 'Retail investors',
        'For companies & financial institutions': 'Companies',
    }[audience]
    view = audience_view(recs, audience_key)
    if not view:
        st.info('No audience-specific notes are available for this view yet.')
    else:
        st.caption('These supplementary notes use the model’s fixed training and July 2025 out-of-sample results; they do not update with the date-range selector above.')
        for i, recommendation in enumerate(view):
            with st.expander(f"{i + 1}. {recommendation['title']}"):
                left, right = st.columns(2)
                with left:
                    st.markdown('**What the analysis found**')
                    st.write(recommendation['finding'])
                    st.markdown('**Why it matters**')
                    st.write(recommendation['why'])
                    st.markdown('**Assets / characteristics involved**')
                    st.write(recommendation['assets'])
                with right:
                    st.markdown('**What to consider**')
                    st.write(recommendation['consider'])
                    st.markdown('**Risks & uncertainties**')
                    st.write(recommendation['risks'])
                    st.markdown('**Supporting metrics**')
                    st.write(recommendation['supporting'])
                st.caption(f"Audience focus: {recommendation['audience']}")

with st.expander('How to read these trends and recommendations'):
    st.markdown('''
    - The date-range analysis uses fixed model weights against historical adjusted daily closes; it does not re-optimise the portfolios for each range.
    - Market-data providers may delay, revise, or temporarily omit observations. “Latest” means the most recent date returned by the provider, not necessarily today's intraday market.
    - The academic reference and supplementary model notes retain their fixed training and July 2025 out-of-sample windows.
    - Historical performance and risk estimates can differ materially from future results and are not trading signals.
    ''')

with st.expander('Simple glossary'):
    for term, definition in GLOSSARY.items():
        st.markdown(f'- **{term}** — {definition}')

disclaimer_box()
