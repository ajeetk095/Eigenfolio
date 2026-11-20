"""Page 7 — Methodology and links to the original research materials."""
from __future__ import annotations

import streamlit as st

from src import config
from src.ui_utils import disclaimer_box

st.set_page_config(page_title='Methodology', page_icon='📚', layout='wide')
st.title('📚 Methodology & Project Research')
st.caption('Plain-language walkthrough, with the original project materials linked below. '
           'Published results are clearly separated from current market monitoring.')

st.header('Research materials')
st.markdown('The repository already contains the project report PDFs and original analysis notebook. '
            'Open or download them using these source links:')
repo = 'https://github.com/ajeetk095/Eigenfolio'
materials = [
    ('Technical report (PDF)', f'{repo}/blob/main/reports/GWP3_Technical_Report.pdf'),
    ('Project explanation (PDF)', f'{repo}/blob/main/reports/GWP3_Project_Explanation.pdf'),
    ('Project explanation (Markdown)', f'{repo}/blob/main/reports/PROJECT_EXPLANATION.md'),
    ('Original complete analysis notebook', f'{repo}/blob/main/notebooks/GWP3_Complete_Analysis.ipynb'),
    ('User guide', f'{repo}/blob/main/reports/USER_GUIDE.md'),
    ('Deployment guide', f'{repo}/blob/main/reports/DEPLOYMENT_GUIDE.md'),
]
for label, url in materials:
    st.markdown(f'- [{label}]({url})')
st.info('The repository does not currently contain a separate assignment-instructions file. '
        'To publish the official instructions, add the authorized document under '
        '`research/assignment/` and link it here; do not redistribute restricted course material.')

st.header('Assignment and scope')
st.markdown(f'''
The original GWP3 exercise uses at least 10 assets and daily prices. Training is
**{config.TRAIN_START} to {config.TRAIN_END}**; the required out-of-sample window is
**{config.TEST_START} to {config.TEST_END}**. The baseline uses maximum-Sharpe mean-variance
optimization, long-only positions, and a 15% per-asset cap. The assignment-period
analysis remains fixed so its results stay reproducible.

The separate live-market views can display history from **2024-01-02 through the
latest available provider date**. Applying fixed GWP3 weights over that period is
historical context only; it is not a refit, forecast, or real-time trading signal.
''')

st.header('Model methodology')
st.markdown('''
1. **Baseline portfolio:** maximum-Sharpe mean-variance optimization under the assignment constraints.
2. **Covariance denoising:** Ledoit–Wolf shrinkage stabilizes covariance estimates.
3. **Diversification cross-check:** K-Medoids clusters assets by correlation-derived distance.
4. **Combination:** shrinkage, clustering, and portfolio optimization are applied in sequence.
5. **Evaluation:** fixed July 2025 out-of-sample test and a separate walk-forward backtest.

All market data is delayed/end-of-day and provider availability is not guaranteed. Missing data is not fabricated.
''')

st.header('Published report results')
rows = []
for model, periods in config.REPORT_RESULTS.items():
    train, test = periods['train'], periods['test']
    rows.append({'Method': model, 'Train Sharpe': train['Sharpe'],
                 'Train annual return': train['Ann. Return'],
                 'July 2025 OOS Sharpe': test['Sharpe'],
                 'July 2025 OOS annual return': test['Ann. Return'],
                 'OOS max drawdown': test['Max Drawdown']})
st.dataframe(__import__('pandas').DataFrame(rows).style.format({
    'Train Sharpe': '{:.3f}', 'Train annual return': '{:.2%}',
    'July 2025 OOS Sharpe': '{:.3f}', 'July 2025 OOS annual return': '{:.2%}',
    'OOS max drawdown': '{:.2%}'}), width='stretch', hide_index=True)
st.caption('Static figures transcribed from the submitted report; not live calculations.')

st.header('Assumptions and limitations')
st.markdown('''
- Historical averages do not predict future returns.
- The required out-of-sample test is only one month and one market regime.
- Taxes, slippage, transaction costs, and turnover are not modeled.
- Yahoo Finance is a best-effort data source; prices may be delayed or unavailable.
''')
disclaimer_box()
