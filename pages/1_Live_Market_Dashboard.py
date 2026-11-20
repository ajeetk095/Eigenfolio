"""Page 1 — live market dashboard and historical exploration."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src import config, visualizations as viz
from src.data_loader import DataFetchError, fetch_latest_quotes, fetch_price_history
from src.data_validation import compute_returns
from src.ui_utils import disclaimer_box, refresh_button

st.set_page_config(page_title='Live Market Dashboard', page_icon='🛰️', layout='wide')
st.title('🛰️ Live Market Dashboard')
st.caption('Delayed/end-of-day market monitoring. Academic assignment results remain '
           'separate and use their original fixed training and test dates.')

refresh = refresh_button()
with st.spinner('Loading historical market prices…'):
    try:
        history = fetch_price_history(
            config.TICKERS, '2024-01-02', date.today().isoformat(),
            force_refresh=refresh)
        prices = history['prices'].reindex(columns=config.TICKERS).ffill().dropna(how='all')
        if prices.empty:
            raise DataFetchError('No historical price rows were returned.')
    except Exception as exc:  # noqa: BLE001
        st.error(f'Historical market data is unavailable: {exc}')
        st.info('Yahoo Finance is a best-effort public source and may rate-limit requests. '
                'Please retry later; unavailable prices are never fabricated.')
        st.stop()

quotes = None
try:
    quotes = fetch_latest_quotes(config.TICKERS, force_refresh=refresh)
except Exception as exc:  # noqa: BLE001
    st.warning(f'Latest quote snapshot unavailable: {exc} Historical charts remain available.')

st.caption(f"Source: {history.get('source', config.DATA_SOURCE_LABEL)} · Retrieved "
           f"{history.get('fetched_at', 'unknown')} UTC · History: "
           f"{prices.index[0].date()} to {prices.index[-1].date()} · "
           'Prices are delayed/end-of-day, not tick-by-tick real-time.')

if quotes is not None:
    quote_frame = quotes['quotes'].set_index('Ticker').reindex(config.TICKERS).rename_axis('Ticker').reset_index()
    st.subheader('Latest available quotes')
    quote_display = quote_frame.copy()
    quote_display['Last Price'] = quote_display['Last Price'].map(lambda x: f'{x:,.2f}' if pd.notna(x) else '—')
    quote_display['Change %'] = quote_display['Change %'].map(lambda x: f'{x:+.2f}%' if pd.notna(x) else '—')
    st.dataframe(quote_display, hide_index=True, width='stretch')
    if quotes.get('failed_tickers'):
        st.caption('Unavailable quote symbols: ' + ', '.join(quotes['failed_tickers']))

st.subheader('Explore history')
left, right = st.columns([1, 2])
with left:
    selected_tickers = st.multiselect('Assets', config.TICKERS, default=config.TICKERS[:6], key='market_assets')
with right:
    date_range = st.date_input(
        'Date range', value=(prices.index[0].date(), prices.index[-1].date()),
        min_value=prices.index[0].date(), max_value=prices.index[-1].date(),
        key='market_date_range')
if not selected_tickers:
    st.info('Choose one or more assets to view charts.')
    st.stop()
if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info('Select both a start date and an end date.')
    st.stop()
start_date, end_date = date_range
view = prices.loc[(prices.index.date >= start_date) & (prices.index.date <= end_date), selected_tickers]
if len(view) < 2:
    st.info('Choose a range containing at least two trading days.')
    st.stop()

chart_type = st.radio('Price view', ['Adjusted price', 'Indexed to 100'], horizontal=True)
if chart_type == 'Indexed to 100':
    st.plotly_chart(viz.normalized_price_chart(view, selected_tickers), width='stretch', key='market_normalized')
else:
    st.plotly_chart(viz.price_chart(view, selected_tickers), width='stretch', key='market_prices')

returns = compute_returns(view)['log']
if returns.empty:
    st.info('Not enough observations in the chosen range to calculate returns.')
    st.stop()
if quotes is not None:
    q = quotes['quotes'].set_index('Ticker').reindex(selected_tickers)
    st.metric('Selected assets up on latest quote session',
              f"{int((q['Change %'] > 0).sum())} / {int(q['Change %'].notna().sum())}")
st.plotly_chart(viz.correlation_heatmap(returns.corr(), 'Correlation — selected assets and dates'),
                width='stretch', key='market_correlation')

trend = pd.DataFrame({
    'Annualized return (period)': returns.mean() * 252,
    'Annualized volatility': returns.std() * 252 ** 0.5,
    'Latest price': view.iloc[-1],
    'Period change %': (view.iloc[-1] / view.iloc[0] - 1) * 100,
})
trend['Trend'] = trend['Period change %'].map(lambda x: '▲ Up' if x > 2 else ('▼ Down' if x < -2 else '◆ Flat'))
st.dataframe(trend.style.format({
    'Annualized return (period)': '{:.1%}', 'Annualized volatility': '{:.1%}',
    'Latest price': '{:,.2f}', 'Period change %': '{:+.1f}%'}), width='stretch')

with st.expander('Download selected historical prices'):
    st.download_button('Download CSV', view.to_csv().encode('utf-8'),
                       file_name='eigenfolio_market_history.csv', mime='text/csv')
disclaimer_box()
