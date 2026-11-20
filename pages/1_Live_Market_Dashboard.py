"""Page 1 — Live Market Dashboard (Environment B: live monitoring)."""
from __future__ import annotations

import streamlit as st

from src import config
from src.data_loader import fetch_latest_quotes, fetch_price_history
from src.data_validation import compute_returns
from src.ui_utils import data_freshness_bar, disclaimer_box, refresh_button
from src import visualizations as viz

st.set_page_config(page_title='Live Market Dashboard', page_icon='🛰️', layout='wide')

st.title('🛰️ Live Market Dashboard')
st.caption('**Environment B — Live monitoring.** This page uses the latest '
           'available market data and is kept fully separate from the academic '
           'GWP3 validation results (Environment A).')

refresh = refresh_button()

try:
    quotes_payload = fetch_latest_quotes(config.TICKERS, force_refresh=refresh)
    q = quotes_payload['quotes'].set_index('Ticker').loc[config.TICKERS].reset_index()
    hist = fetch_price_history(config.TICKERS,
                               config.TRAIN_START, config.TEST_END,
                               force_refresh=refresh)
    prices = hist['prices'].reindex(columns=config.TICKERS).ffill().dropna()
except Exception as exc:                       # noqa: BLE001
    st.error(f'Market data unavailable: {exc}')
    st.stop()

data_freshness_bar({**hist, 'validation': {'first_date': str(prices.index[0].date()),
                                           'last_date': str(prices.index[-1].date()),
                                           'n_days': len(prices)}})

if quotes_payload['failed_tickers']:
    st.warning(f"No live quote for: {', '.join(quotes_payload['failed_tickers'])} "
               '(historical data still shown for them).')

# ---------------------------------------------------------------- quotes ----
st.subheader('Latest available prices')
q_disp = q.copy()
q_disp['Last Price'] = q_disp['Last Price'].map('{:,.2f}'.format)
q_disp['Change %'] = q_disp['Change %'].map(lambda x: f'{x:+.2f}%' if x == x else '—')
st.dataframe(q_disp, width='stretch', hide_index=True)

st.caption(f"Source: {quotes_payload['source']} · Snapshot: "
           f"{quotes_payload['fetched_at']} UTC. Prices are delayed/end-of-day; "
           'they are not tick-by-tick real-time quotes.')

# ------------------------------------------------------------- selection ----
st.subheader('Historical exploration')
sel = st.multiselect('Select assets', config.TICKERS,
                     default=config.TICKERS[:6])
date_min, date_max = prices.index[0].date(), prices.index[-1].date()
rng = st.slider('Date range', min_value=date_min, max_value=date_max,
                value=(date_min, date_max))
if not sel:
    st.info('Select at least one asset to display charts.')
    st.stop()

view = prices.loc[(prices.index >= str(rng[0])) & (prices.index <= str(rng[1])), sel]
st.plotly_chart(viz.price_chart(view, sel), width='stretch')

rets = compute_returns(view)['log']
c1, c2 = st.columns(2)
with c1:
    st.metric('Assets up on latest day', f"{int((q.set_index('Ticker').loc[sel, 'Change %'] > 0).sum())} / {len(sel)}")
with c2:
    ann_vol = rets.std() * (252 ** 0.5)
    st.write('**Annualised volatility (selected window)**')
    st.dataframe(ann_vol.map('{:.1%}'.format).rename('Ann. Vol').to_frame().T,
                 width='stretch')

st.plotly_chart(viz.correlation_heatmap(rets.corr(),
                                        'Correlation — selected window'),
                width='stretch')

# trend summary
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
