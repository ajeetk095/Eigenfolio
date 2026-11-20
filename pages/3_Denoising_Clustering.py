"""Page 3 — Denoising and Clustering: the ML machinery, visualized."""
from __future__ import annotations

import numpy as np
import streamlit as st

from src import config
from src.ui_utils import data_freshness_bar, disclaimer_box, get_result, refresh_button
from src import visualizations as viz

st.set_page_config(page_title='Denoising & Clustering', page_icon='🧠', layout='wide')
st.title('🧠 Denoising & Clustering — Machine Learning Improvements')

refresh = refresh_button()
try:
    result = get_result(force_refresh=refresh)
except Exception as exc:                       # noqa: BLE001
    st.error(f'Market data unavailable: {exc}')
    st.stop()

data_freshness_bar(result)
dn = result['denoise']
cl = result['clusters']

# ------------------------------------------------------------- denoising ----
st.header('Step 2 — Denoising via Ledoit-Wolf covariance shrinkage')
st.markdown("""
**The problem.** Mean-variance optimization needs the covariance matrix **Σ** —
but we can only *estimate* it from a finite sample (373 trading days of 12 assets).
Sample estimates contain noise: extreme eigenvalues that, when the matrix is
inverted during optimization, get amplified into unstable, over-concentrated weights.

**The fix.** Ledoit & Wolf (2004) shrink the sample covariance toward a
structured target (the average variance × identity):

$$\\hat{\\Sigma}_{LW} = (1-\\alpha^*)\\,\\hat{\\Sigma} + \\alpha^* \\cdot \\frac{\\mathrm{tr}(\\hat{\\Sigma})}{n}\\,I$$

where α* is computed **analytically** (no cross-validation) to minimize the
expected squared estimation error. Shrinkage pulls extreme eigenvalues toward
the average, making matrix inversion more stable.
""")

d1, d2, d3, d4 = st.columns(4)
d1.metric('Shrinkage intensity α*', f"{dn['alpha']:.4f}",
          help='4.4% toward the structured target in the submitted report — '
               'consistent with a data-rich regime T/n ≈ 31.')
d2.metric('Condition number (sample)', f"{dn['cond_sample']:.1f}")
d3.metric('Condition number (LW)', f"{dn['cond_lw']:.1f}")
d4.metric('Improvement', f"{dn['cond_improvement']:.2f}×",
          help='Report published 55.8 → 39.5 (1.4×). Small differences vs. '
               'the report come from the newer data window.')

with st.expander('🔍 What does "condition number" mean?'):
    st.markdown('The ratio of the largest to the smallest eigenvalue. A high '
                'condition number means the matrix is close to being '
                'ill-behaved when inverted — small input errors become large '
                'weight errors. Lower = more numerically stable, more '
                'reproducible weights.')

st.plotly_chart(viz.eigenvalue_chart(dn['eig_sample'], dn['eig_lw']),
                width='stretch')

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(viz.correlation_heatmap(
        dn['corr_sample'], 'Sample correlation (training period)'),
        width='stretch')
with c2:
    st.plotly_chart(viz.correlation_heatmap(
        dn['corr_lw'], 'Denoised (LW) correlation'),
        width='stretch')

st.info('**Why identical weights?** In the submitted report, denoising alone '
        'produced the *same* weights as the baseline: with only 4.4% shrinkage, '
        'the binding constraint is the 15% per-asset cap, not the covariance '
        'structure. The benefit is hidden but real — better-conditioned '
        'estimates and more stable weights across market environments. For '
        'larger universes (n > 50, T/n < 10) shrinkage becomes decisive.')
st.markdown('**When denoising may NOT help:** very data-rich, well-conditioned '
            'samples (α* → 0 changes nothing) — and it never fixes a wrong '
            'expected-return estimate μ, which shrinkage deliberately leaves '
            'untouched.')

# ------------------------------------------------------------ clustering ----
st.header('Step 2/3 — Clustering via K-Medoids')
st.markdown("""
Assets are grouped by a correlation-derived distance
$d(i,j) = \\sqrt{(1-\\rho_{ij})/2} \\in [0,1]$, then clustered with **K-Medoids**
(K = 4). K-Medoids uses actual data points as cluster centres, making it robust
to outliers — a known issue with financial data. Within each cluster, assets
receive **inverse-volatility risk-parity weights**; each cluster gets an equal
25% budget, then the 15% cap is applied.
""")

cluster_rows = []
for c, info in sorted(cl['clusters'].items()):
    cluster_rows.append({
        'Cluster': f'C{c + 1}',
        'Medoid': info['medoid'],
        'Size': len(info['members']),
        'Members': ', '.join(info['members']),
    })
st.dataframe(cluster_rows, width='stretch', hide_index=True)

st.plotly_chart(viz.cluster_heatmap(cl['corr'], cl['labels'], config.TICKERS),
                width='stretch')

st.markdown(f"""
**What the clusters tell us.** In the submitted report, 9 of 12 assets fell
into one technology cluster (C1), with TSLA, PEP and COST each isolated —
the universe is dominated by a common technology factor. That is exactly why
the pure cluster portfolio is the most *balanced* (lowest concentration) but
not the highest-returning: clustering ignores expected returns entirely.
""")
st.markdown('**When clustering may NOT help:** when cluster structure is weak '
            'or unstable across windows (as here, with one dominant cluster), '
            'or when return information matters more than diversification.')

# -------------------------------------------------------- combination -------
st.header('Step 3 — Combination pipeline: Denoise → Cluster → MVO')
st.markdown("""
1. **Denoise** — LW shrinkage stabilizes Σ (cleaner correlations for clustering)
2. **Cluster** — K-Medoids groups correlated assets, reducing the effective
   problem from 12 assets to 4 clusters (T/n from ≈31 to ≈93)
3. **Optimize** — max-Sharpe MVO at cluster level, expanded to assets, 15% cap
""")
st.plotly_chart(viz.cluster_allocation_chart(result['combo']['cluster_alloc']),
                width='stretch')
st.caption('Report reference: cluster-level MVO assigned 47.1% to C1 '
           '(technology) and 52.9% to C4 (COST), 0% to TSLA and PEP clusters.')

disclaimer_box()
