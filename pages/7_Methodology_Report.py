"""Methodology page with report, notebook, and project instructions available in-app."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src import config
from src.ui_utils import disclaimer_box

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / 'reports' / 'GWP3_Technical_Report.pdf'
NOTEBOOK_PATH = ROOT / 'notebooks' / 'GWP3_Complete_Analysis.ipynb'
INSTRUCTIONS_PATH = ROOT / 'research' / 'PROJECT_INSTRUCTIONS.md'

st.set_page_config(page_title='Methodology & Research', page_icon='📚', layout='wide')
st.title('📚 Methodology & Project Research')
st.caption('Browse the project documents in this app. The original academic windows '
           'are preserved separately from the live-market context.')

st.header('Project documents')
doc_tabs = st.tabs(['📄 Technical report', '📓 Original notebook', '🧭 Project instructions'])

with doc_tabs[0]:
    if REPORT_PATH.is_file():
        report_bytes = REPORT_PATH.read_bytes()
        st.download_button('Download technical report (PDF)', report_bytes,
                           file_name=REPORT_PATH.name, mime='application/pdf', key='download_technical_report')
        encoded = base64.b64encode(report_bytes).decode('ascii')
        components.html(
            f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="760" '
            'style="border:1px solid #d5dbe3;border-radius:8px"></iframe>',
            height=780, scrolling=True)
    else:
        st.warning('Technical report PDF is not present in this deployment package.')

with doc_tabs[1]:
    if NOTEBOOK_PATH.is_file():
        notebook_bytes = NOTEBOOK_PATH.read_bytes()
        st.download_button('Download original Jupyter notebook', notebook_bytes,
                           file_name=NOTEBOOK_PATH.name, mime='application/x-ipynb+json', key='download_original_notebook')
        try:
            notebook = json.loads(notebook_bytes)
            code_cells = 0
            for number, cell in enumerate(notebook.get('cells', []), start=1):
                source = ''.join(cell.get('source', []))
                if not source.strip():
                    continue
                if cell.get('cell_type') == 'markdown':
                    with st.expander(f'Notebook cell {number} · Markdown'):
                        st.markdown(source)
                else:
                    code_cells += 1
                    with st.expander(f'Notebook cell {number} · Code'):
                        st.code(source, language='python')
            st.caption(f'Displaying non-empty notebook cells in original order ({code_cells} code cells).')
        except (json.JSONDecodeError, OSError):
            st.error('The notebook could not be parsed for display; use the download button to open the original file.')
    else:
        st.warning('Original analysis notebook is not present in this deployment package.')

with doc_tabs[2]:
    if INSTRUCTIONS_PATH.is_file():
        st.markdown(INSTRUCTIONS_PATH.read_text(encoding='utf-8'))
        st.download_button('Download project instructions', INSTRUCTIONS_PATH.read_bytes(),
                           file_name=INSTRUCTIONS_PATH.name, mime='text/markdown', key='download_project_instructions')
    else:
        st.info('No project instructions document was bundled with this release.')

st.header('Academic scope and reproducibility')
st.markdown(f'''
The original GWP3 training period is **{config.TRAIN_START} to {config.TRAIN_END}**;
the required out-of-sample period is **{config.TEST_START} to {config.TEST_END}**.
Those fixed dates and constraints are retained for reproducibility. The separate
market dashboard may show history from 2024-01-02 through the latest provider date;
that context is not a refit or forecast.
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
st.caption('Published static figures from the report; distinct from current market calculations.')

disclaimer_box()
