"""Page 1 — Live Market Dashboard (Environment B: live monitoring)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import DataFetchError, fetch_latest_quotes, fetch_price_history
from src.data_validation import compute_returns
from src.ui_utils import data_freshness_bar, disclaimer_box, refresh_button
from src import visualizations as viz

st.set_page_config(page_title='Live Market Dashboard', page_icon='🛰️', layout='wide')

st.title('🛰️ Live Market Dashboard')
st.caption('**Environment B — Market monitoring.** Prices are delayed/end-of-day, '
           'not tick-by-tick real-time, and are separate from academic validation.')

refresh = refresh_button()

# Keep historical exploration available if only the quote endpoint is unavailable.
try:
    hist = fetch_price_history(config.TICKERS, config.TRAIN_START,
                               config.TEST_END, force_refresh=refresh)
    prices = hist['prices'].reindex(columns=config.TICKERS).ffill().dropna()
    if prices.empty:
        raise DataFetchError('The provider returned no complete historical rows.')
except Exception as exc:                       # noqa: BLE001
    st.error(f'Historical market data unavailable: {exc}')
    st.info('Check your connection and retry. Yahoo Finance can temporarily rate-limit '
            'requests; no synthetic or stale prices are substituted.')
    st.stop()

try:
    quotes_payload = fetch_latest_quotes(config.TICKERS, force_refresh=refresh)
    q = quotes_payload['quotes'].set_index('Ticker').reindex(config.TICKERS)
    q.index.name = 'Ticker'
    q = q.reset_index()
except Exception as exc:                       # noqa: BLE001
    quotes_payload = None
    q = pd.DataFrame(columns=['Ticker', 'Last Price', 'Prev Close', 'Change %'])
    st.warning(f'Latest quotes are temporarily unavailable: {exc} '
               'Historical charts below are still usable.')

# data_freshness_bar expects `data_source`; the downloader returns `source`.
# Map the name here and keep the helper's model-result contract explicit.
validation = {
    'first_date': str(prices.index[0].date()),
    'last_date': str(prices.index[-1].date()),
    'n_days': len(prices),
}
data_freshness_bar({
    'data_source': hist.get('source', config.DATA_SOURCE_LABEL),
    'fetched_at': hist.get('fetched_at', 'unknown'),
    'validation': validation,
})

if quotes_payload is not None:
    if quotes_payload['failed_tickers']:
        st.warning(f"No latest quote for: {', '.join(quotes_payload['failed_tickers'])} "
                   '(historical prices are still shown where available).')
    st.subheader('Latest available prices')
    q_disp = q.copy()
    q_disp['Last Price'] = q_disp['Last Price'].map(
        lambda x: f'{x:,.2f}' if pd.notna(x) else '—')
    q_disp['Change %'] = q_disp['Change %'].map(
        lambda x: f'{x:+.2f}%' if pd.notna(x) else '—')
    st.dataframe(q_disp, width='stretch', hide_index=True)
    st.caption(f"Source: {quotes_payload['source']} · Snapshot: "
               f"{quotes_payload['fetched_at']} UTC. Delayed/end-of-day data.")

st.subheader('Historical exploration')
sel = st.multiselect('Select assets', config.TICKERS, default=config.TICKERS[:6])
date_min, date_max = prices.index[0].date(), prices.index[-1].date()
rng = st.slider('Date range', min_value=date_min, max_value=date_max,
                value=(date_min, date_max))
if not sel:
    st.info('Select at least one asset to display charts.')
    st.stop()

view = prices.loc[(prices.index.date >= rng[0]) & (prices.index.date <= rng[1]), sel]
if len(view) < 2:
    st.info('Choose a date range containing at least two trading days.')
    st.stop()
st.plotly_chart(viz.price_chart(view, sel), width='stretch')

rets = compute_returns(view)['log']
c1, c2 = st.columns(2)
with c1:
    if quotes_payload is not None:
        latest = q.set_index('Ticker').reindex(sel)['Change %']
        st.metric('Assets up on latest session',
                  f"{int((latest > 0).sum())} / {int(latest.notna().sum())}")
    else:
        st.metric('Assets selected', str(len(sel)))
with c2:
    ann_vol = rets.std() * (252 ** 0.5)
    st.write('**Annualised volatility (selected window)**')
    st.dataframe(ann_vol.map('{:.1%}'.format).rename('Ann. Vol').to_frame().T,
                 width='stretch')

st.plotly_chart(viz.correlation_heatmap(rets.corr(),
                                        'Correlation — selected window'),
                width='stretch')

st.subheader('Market trend summary')
trend = pd.DataFrame({
    'Ann. return (window)': rets.mean() * 252,
    'Ann. volatility': rets.std() * 252 ** 0.5,
    'Latest price': view.iloc[-1],
    'Window change %': (view.iloc[-1] / view.iloc[0] - 1) * 100,
})
trend['Trend'] = trend['Window change %'].map(
    lambda x: '▲ Up' if x > 2 else ('▼ Down' if x < -2 else '◆ Flat'))
st.dataframe(trend.style.format({
    'Ann. return (window)': '{:.1%}', 'Ann. volatility': '{:.1%}',
    'Latest price': '{:,.2f}', 'Window change %': '{:+.1f}'}),
    width='stretch')

disclaimer_box()
