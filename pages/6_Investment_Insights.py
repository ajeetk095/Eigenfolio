"""Page 6 — Investment Insights & Recommendations (intelligent insights engine)."""
from __future__ import annotations

import streamlit as st

from src.recommendations import audience_view, generate_recommendations
from src.ui_utils import data_freshness_bar, disclaimer_box, get_result, refresh_button

st.set_page_config(page_title='Investment Insights', page_icon='💡', layout='wide')
st.title('💡 Investment Insights & Recommendations')
st.caption('Every insight below is generated dynamically from the computed '
           'model outputs (weights, metrics, clustering, risk contributions). '
           'Nothing is hardcoded, and nothing here is a claim about future '
           'prices or guaranteed profits.')

refresh = refresh_button()
try:
    result = get_result(force_refresh=refresh)
except Exception as exc:                       # noqa: BLE001
    st.error(f'Market data unavailable: {exc}')
    st.stop()

data_freshness_bar(result)

audience = st.radio('Choose your view',
                    ['For investors', 'For retail investors', 'For companies & financial institutions'],
                    horizontal=True)

recs = generate_recommendations(result)
key = {'For investors': 'Investors',
       'For retail investors': 'Retail investors',
       'For companies & financial institutions': 'Companies'}[audience]
view = audience_view(recs, key)

if not view:
    st.info('No audience-specific recommendations available for this view.')
    st.stop()

for i, r in enumerate(view):
    with st.expander(f"**{i + 1}. {r['title']}**", expanded=(i == 0)):
        rows = [
            ('1. What the analysis found', r['finding']),
            ('2. Why it matters', r['why']),
            ('3. Assets / characteristics involved', r['assets']),
            ('4. What to consider', r['consider']),
            ('5. Risks & uncertainties remaining', r['risks']),
            ('6. Metrics supporting the conclusion', r['supporting']),
        ]
        for title, body in rows:
            st.markdown(f'**{title}**  \n{body}')
        st.caption(f'Audience focus: {r["audience"]}')

st.markdown('### How to read these recommendations')
st.markdown("""
- Recommendations describe the **estimated, historical behaviour** of the model
  portfolios — they are research observations, not trading signals.
- Where the engine says a portfolio “offered the best historical risk-adjusted
  return”, it refers strictly to the training window (Jan 2024 – Jun 2025).
- Out-of-sample (July 2025) results can — and in the submitted report, did —
  contradict training-period rankings. Both are always shown.
""")

st.markdown('### Simple glossary')
from src.ui_utils import GLOSSARY
for term, definition in GLOSSARY.items():
    st.markdown(f'- **{term}** — {definition}')

disclaimer_box()
