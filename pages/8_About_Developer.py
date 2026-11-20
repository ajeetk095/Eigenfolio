"""Page 8 — About the Developer / Hire Me."""
from __future__ import annotations

import streamlit as st

from src.config import DEVELOPER

st.set_page_config(page_title='About the Developer', page_icon='👤', layout='wide')

st.markdown(
    f"""
    <div style="padding:2rem;border-radius:14px;text-align:center;
    background:linear-gradient(120deg,#0d1b2a,#1b4965 55%,#2c7da0);">
      <h1 style="color:#ffffff;margin:0;">{DEVELOPER['headline']}</h1>
      <p style="color:#dbe9f4;font-size:1.1rem;margin-top:0.6rem;">
        Bridging quantitative research and production software.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('---')

c1, c2 = st.columns([2, 1])
with c1:
    st.header('👋 About me')
    st.markdown(DEVELOPER['intro'])
    st.markdown(f"**Career focus:** {DEVELOPER['career_focus']}")
with c2:
    st.header('🏆 Achievements')
    for a in DEVELOPER['achievements']:
        st.markdown(f'- {a}')

st.header('🛠️ Professional skills')
skills = DEVELOPER['skills']
cols = st.columns(4)
for i, skill in enumerate(skills):
    with cols[i % 4]:
        st.markdown(f'<div style="padding:0.55rem 0.9rem;border-radius:10px;'
                    f'background:#eaf2f8;text-align:center;font-weight:600;">'
                    f'{skill}</div>', unsafe_allow_html=True)

st.markdown('---')
st.header('This project')
st.markdown("""
**Quant Portfolio Intelligence** is my MScFE 652 Group Work Project 3
(WorldQuant University) turned into a deployable, interactive platform:
mean-variance optimization, Ledoit-Wolf covariance denoising, K-Medoids
clustering, walk-forward backtesting, live market-data integration, a dynamic
recommendations engine, tested financial calculations and a security review —
all in a modular, documented Streamlit codebase.
""")

st.header('📬 Get in touch')
b1, b2, b3, b4 = st.columns(4)
b1.link_button('💼 Hire the Developer',
               f"mailto:{DEVELOPER['email']}?subject=Hiring%20Enquiry%20—%20"
               'Quant%20Research%20%2F%20Financial%20Engineering%20Role')
b2.link_button('🔗 Connect on LinkedIn', DEVELOPER['linkedin'])
b3.link_button('🐙 Explore GitHub Projects', DEVELOPER['github'])
b4.link_button('✉️ Contact by Email',
               f"mailto:{DEVELOPER['email']}?subject=Quant%20Portfolio%20"
               'Intelligence%20—%20Contact')

st.markdown('---')
st.caption(f"GitHub: {DEVELOPER['github']} · LinkedIn: {DEVELOPER['linkedin']} · "
           f"Email: {DEVELOPER['email']}")
st.info('All profile information above is provided by the developer. This page '
        'does not assert any results, awards or employment beyond the listed items.')
