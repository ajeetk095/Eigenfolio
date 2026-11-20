"""Page 2 — Portfolio Optimization Lab with interactive allocation views."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from src import config, visualizations as viz
from src.data_loader import fetch_price_history
from src.data_validation import compute_returns
from src.pipeline import metrics_table
from src.ui_utils import constraint_check_pills, data_freshness_bar, disclaimer_box, get_result, refresh_button, weights_dict

st.set_page_config(page_title='Portfolio Optimization', page_icon='⚖️', layout='wide')
st.title('⚖️ Portfolio Optimization Lab')
st.caption('Official GWP3 model weights retain the assignment’s fixed training window '
           'and constraints. The live-period chart evaluates those fixed weights on '
           'market history through today; it does not refit or imply a forecast.')

refresh = refresh_button()
try:
    result = get_result(force_refresh=refresh)
except Exception as exc:  # noqa: BLE001
    st.error(f'Academic model results unavailable: {exc}')
    st.info('The assignment results require complete Jan 2024–Jul 2025 data. Retry when the provider is available.')
    st.stop()

data_freshness_bar(result)
constraint_check_pills(result)
methods = list(result['weights'])

st.subheader('Portfolio allocation — interactive')
controls = st.columns([1.2, 1, 1])
with controls[0]:
    selected_methods = st.multiselect('Models to display', methods, default=methods, key='opt_methods')
with controls[1]:
    selected_assets = st.multiselect('Assets to display', config.TICKERS, default=config.TICKERS, key='opt_assets')
with controls[2]:
    min_weight_pct = st.slider('Hide allocations below', min_value=0.0, max_value=10.0,
                               value=0.0, step=0.25, format='%.2f%%')
if not selected_methods:
    st.info('Select at least one model to display its allocation.')
    st.stop()
if not selected_assets:
    st.info('Select one or more assets; clear the minimum-weight threshold to see small positions.')
    st.stop()

method = st.selectbox('Allocation chart model', selected_methods, index=0, key='allocation_model')
all_weights = weights_dict(result, method)
visible = {ticker: weight for ticker, weight in all_weights.items()
           if ticker in selected_assets and weight * 100 >= min_weight_pct}
if not visible:
    st.warning('No positions match the current asset and minimum-weight filters.')
else:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.allocation_pie(visible, f'{method} — selected allocation'),
                        width='stretch', key='allocation_pie')
    with c2:
        allocation_frame = pd.DataFrame({'Ticker': list(visible), 'Weight (%)': [w * 100 for w in visible.values()]})
        fig = viz.px.bar(allocation_frame, x='Ticker', y='Weight (%)', color='Ticker',
                         title=f'{method} — individual holdings', text_auto='.2f')
        fig.update_layout(showlegend=False, template=viz.APP_TEMPLATE, xaxis_title='Asset', yaxis_title='Weight (%)')
        fig.add_hline(y=config.MAX_WEIGHT * 100, line_dash='dash', line_color='red', annotation_text='15% assignment cap')
        st.plotly_chart(fig, width='stretch', key='allocation_bar')
    shown_total = sum(visible.values())
    st.caption(f'Visible allocation: {shown_total:.2%} of portfolio. Filters affect display only; '
               'the official portfolio constraints and weights are not changed.')

st.subheader('Holdings table')
period = st.radio('Metrics period', ['Training (Jan 2024–Jun 2025)', 'Out-of-sample (Jul 2025)'],
                  horizontal=True, key='opt_metric_period')
metric_period = 'train' if period.startswith('Training') else 'test'
metrics = metrics_table(result['metrics'], metric_period)
rows = []
for model_name in selected_methods:
    weights = weights_dict(result, model_name)
    for ticker, weight in weights.items():
        if ticker not in selected_assets or weight * 100 < min_weight_pct:
            continue
        rows.append({'Model': model_name, 'Ticker': ticker, 'Weight': weight,
                     'Sector': config.SECTORS[ticker], 'Company': config.ASSET_DESCRIPTIONS[ticker],
                     'Ann. Return': metrics.loc[model_name, 'Ann. Return'],
                     'Ann. Vol': metrics.loc[model_name, 'Ann. Vol'],
                     'Sharpe': metrics.loc[model_name, 'Sharpe']})
table = pd.DataFrame(rows)
if not table.empty:
    sort_col = st.selectbox('Sort holdings table by', ['Weight', 'Ticker', 'Model', 'Sharpe'], key='holdings_sort')
    table = table.sort_values(sort_col, ascending=(sort_col == 'Ticker')).reset_index(drop=True)
    st.dataframe(table.style.format({'Weight': '{:.2%}', 'Ann. Return': '{:.2%}',
                                     'Ann. Vol': '{:.2%}', 'Sharpe': '{:.3f}'}),
                 width='stretch', hide_index=True)
    st.download_button('Download filtered holdings CSV', table.to_csv(index=False).encode('utf-8'),
                       file_name='eigenfolio_filtered_holdings.csv', mime='text/csv')
else:
    st.info('No holdings match the selected filters.')

st.subheader('Risk–return map and efficient frontier')
st.plotly_chart(viz.frontier_chart(result['frontier'],
                                   {m: result['weights'][m] for m in selected_methods},
                                   result['mu'], result['S_sample']),
                width='stretch', key='efficient_frontier')
st.caption('Hover, zoom, and use the legend to isolate series. Estimates are historical, not forecasts.')

st.subheader('Historical market context — fixed model weights')
st.caption('Fetches current adjusted prices from 2024-01-02 through today. This is a '
           'historical comparison applying assignment-period weights to subsequent '
           'returns, not a live re-optimization or investment recommendation.')
try:
    live_history = fetch_price_history(config.TICKERS, '2024-01-02', date.today().isoformat(),
                                       force_refresh=refresh)
    live_prices = live_history['prices'].reindex(columns=config.TICKERS).ffill().dropna(how='all')
    if live_prices.empty:
        st.info('No current market history is available from the provider.')
    else:
        start_end = st.date_input('Performance chart dates',
                                  value=(live_prices.index[0].date(), live_prices.index[-1].date()),
                                  min_value=live_prices.index[0].date(), max_value=live_prices.index[-1].date(),
                                  key='performance_dates')
        if isinstance(start_end, (tuple, list)) and len(start_end) == 2:
            start_date, end_date = start_end
            live_prices = live_prices.loc[(live_prices.index.date >= start_date) &
                                          (live_prices.index.date <= end_date)]
        if len(live_prices) >= 2:
            live_returns = compute_returns(live_prices)['log']
            performance = {}
            for model_name in selected_methods:
                w = np.asarray(result['weights'][model_name], dtype=float)
                portfolio_log_returns = live_returns @ w
                performance[model_name] = np.exp(portfolio_log_returns.cumsum())
            st.plotly_chart(viz.cumulative_return_chart(
                performance, 'Historical cumulative performance — fixed GWP3 weights'),
                width='stretch', key='live_model_performance')
            st.caption(f"Prices retrieved {live_history.get('fetched_at', 'unknown')} UTC · "
                       'Results use portfolio weights fixed from the academic training period.')
        else:
            st.info('Choose at least two trading dates for the performance chart.')
except Exception as exc:  # noqa: BLE001
    st.warning(f'Current market context could not be loaded: {exc} Assignment-period model results above remain available.')

with st.expander('Optimization settings and interpretation'):
    st.markdown('''
    | Setting | Value |
    |---|---|
    | Objective | Maximum Sharpe ratio |
    | Official constraints | Fully invested, long-only, maximum 15% per asset |
    | Academic training | Jan 2024 – Jun 2025 |
    | Required academic test | July 2025, kept separate from training |
    | Live-history view | 2024-01-02 through latest provider date; fixed academic weights |
    | Trading costs | Not modeled |
    ''')

disclaimer_box()
