"""Page 8 — About the developer and project."""
from __future__ import annotations

import streamlit as st

from src.config import DEVELOPER
from src.ui_utils import disclaimer_box

st.set_page_config(page_title='About the Developer', page_icon='👤', layout='wide')
st.markdown(
    f'''<div style="padding:2rem;border-radius:14px;text-align:center;
    background:linear-gradient(120deg,#0d1b2a,#1b4965 55%,#2c7da0);">
    <h1 style="color:#ffffff;margin:0;">{DEVELOPER['headline']}</h1>
    <p style="color:#dbe9f4;font-size:1.1rem;margin-top:0.6rem;">Bridging quantitative research and production software.</p>
    </div>''', unsafe_allow_html=True)

c1, c2 = st.columns([2, 1])
with c1:
    st.header('👋 About me')
    st.markdown(DEVELOPER['intro'])
    st.markdown(f"**Career focus:** {DEVELOPER['career_focus']}")
with c2:
    st.header('🏆 Achievements')
    for achievement in DEVELOPER['achievements']:
        st.markdown(f'- {achievement}')

st.header('🛠️ Professional skills')
cols = st.columns(4)
for i, skill in enumerate(DEVELOPER['skills']):
    with cols[i % 4]:
        st.markdown(f'<div style="padding:0.55rem 0.9rem;border-radius:10px;'
                    f'background:#eaf2f8;text-align:center;font-weight:600;">'
                    f'{skill}</div>', unsafe_allow_html=True)

st.header('This project')
st.markdown('''
**Quant Portfolio Intelligence** presents a portfolio-management research project
through an interactive Streamlit application: mean-variance optimization,
Ledoit–Wolf covariance denoising, K-Medoids clustering, walk-forward backtesting,
market data, and a dynamically generated insights engine.
''')
st.header('📬 Get in touch')
b1, b2, b3, b4 = st.columns(4)
b1.link_button('💼 Hire the Developer', f"mailto:{DEVELOPER['email']}?subject=Hiring%20Enquiry")
b2.link_button('🔗 Connect on LinkedIn', DEVELOPER['linkedin'])
b3.link_button('🐙 Explore GitHub Projects', DEVELOPER['github'])
b4.link_button('✉️ Contact by Email', f"mailto:{DEVELOPER['email']}?subject=Project%20Enquiry")
st.caption(f"GitHub: {DEVELOPER['github']} · LinkedIn: {DEVELOPER['linkedin']} · Email: {DEVELOPER['email']}")
disclaimer_box()
