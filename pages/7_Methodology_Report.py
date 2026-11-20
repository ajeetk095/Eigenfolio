"""Page 7 — Methodology & Project Report explained for everyone."""
from __future__ import annotations

import streamlit as st

from src import config
from src.ui_utils import data_freshness_bar, disclaimer_box, get_result, refresh_button

st.set_page_config(page_title='Methodology', page_icon='📚', layout='wide')
st.title('📚 Methodology & Project Report')
st.caption('The complete GWP3 academic project, explained in plain language — '
           'no quantitative-finance background needed.')

refresh = refresh_button()
try:
    result = get_result(force_refresh=refresh)
except Exception as exc:                       # noqa: BLE001
    st.error(f'Market data unavailable: {exc}')
    st.stop()
data_freshness_bar(result)

st.header('1. The original assignment (MScFE 652 GWP3, WorldQuant University)')
st.markdown("""
Build a portfolio of **at least 10 assets** with **daily data**, using
**Jan 1 2024 – Jun 30 2025** as the training period. Step 1: find the optimal
portfolio with **no short selling** and **no asset above 15%**. Step 2: explain
machine-learning improvements (denoising / clustering / backtesting — one per
group member; this was a single-member group, so **denoising** was the chosen
topic, with clustering and backtesting added as cross-checks). Step 3: apply
the improvements. Step 4: compare with at least three metrics. Step 5: test
the final portfolio out-of-sample on **July 1–31, 2025**.
""")

st.header('2. The research problem')
st.markdown("""
Classic portfolio optimization (Markowitz, 1952) needs two inputs nobody can
observe directly: expected returns (μ) and how assets co-move (the covariance
matrix, Σ). Both must be estimated from history — and estimation error flows
straight into the portfolio weights, producing unstable, over-concentrated
allocations that disappoint out of sample. **Question:** can simple machine-
learning adjustments reduce that estimation error and improve diversification
enough to matter?
""")

st.header('3. The dataset')
st.markdown("""
12 large-cap U.S. equities across technology, communication services, consumer
discretionary, consumer staples and semiconductors (AAPL, MSFT, GOOGL, AMZN,
NVDA, META, TSLA, AVGO, COST, ASML, PEP, NFLX). Daily adjusted close prices
from Yahoo Finance; 373 training days; 21 out-of-sample days. Daily log
returns; annualisation by 252 trading days; risk-free rate 5.25% p.a.
""")

st.header('4. The method, step by step (in plain language)')
st.markdown(f"""
- **Step 1 — Baseline.** Pick the mix of the 12 stocks with the best
  risk-adjusted historical return, under the two hard rules (never short,
  never above 15% in one name).
- **Step 2 — Denoising.** The table of “how assets move together” is noisy.
  Ledoit-Wolf shrinkage blends it slightly with a simpler, stable table — the
  optimal blend α* is computed analytically. In the report α* = 4.4%, and the
  numerical condition of the matrix improved from 55.8 to 39.5 (1.4×).
- **Step 2 (cross-check) — Clustering.** K-Medoids groups stocks that move
  alike (K = 4): nine tech-heavy names in one cluster; TSLA, PEP and COST
  standing apart. Each cluster gets inverse-volatility weights and a 25%
  budget.
- **Step 2 (cross-check) — Backtesting.** A walk-forward simulation: re-fit
  every month using only the previous ~6 months, judge on the next unseen
  month — the honest way to test a strategy.
- **Step 3 — Combination.** Denoise → Cluster → Optimize, in that order:
  cleaning first improves the grouping; grouping shrinks the problem from 12
  assets to 4 clusters; optimization sets the final cluster mix.
""")

st.header('5. The results (as published in the submitted report)')
st.markdown("""
| Method | Train Sharpe | Train ann. return | OOS (Jul 2025) ann. return | OOS Sharpe | OOS max DD |
|---|---|---|---|---|---|
| Baseline (MVO) | 1.572 | 44.23% | 12.85% | 0.582 | −2.13% |
| Denoise (LW) | 1.572 | 44.23% | 12.85% | 0.582 | −2.13% |
| Cluster (K-Med) | 0.573 | 18.92% | 10.07% | 0.353 | −1.84% |
| Combo (D+C) | 1.231 | 33.17% | **−7.29%** | −1.318 | **−1.65%** (best) |

CVaR 95% OOS: Combo −17.67% (best tail risk) vs Baseline −24.69%.
""")
st.warning('**The honest headline:** the ML-enhanced combination portfolio was '
           'the WORST return performer in the required out-of-sample month. '
           'July 2025 rewarded concentrated tech momentum; the combination '
           'held 24.1% in defensive COST and had exited PEP. One month cannot '
           'validate or refute a structural advantage — the report says exactly '
           'that, and this app preserves that conclusion.')
st.info('The tables above are the static published report figures (computed on '
        'data downloaded Jan–Jul 2025). The other pages show freshly recomputed '
        'values from today\u2019s data; small differences are expected and '
        'disclosed — never silently mixed.')

st.header('6. What the numbers mean')
st.markdown("""
- **The Sharpe collapse (1.572 → 0.582)** is the classic overfitting lesson:
  models fitted to a bull run look brilliant until they meet new data.
- **Denoising changed no weights** here — with only 4.4% shrinkage the 15% cap
  binds first — yet it remains a free, always-on safeguard for numerical
  stability, and it matters far more in bigger universes.
- **Clustering bought drawdown protection** (−1.84% vs −2.13% OOS max DD) at
  the price of returns: it ignores expected returns entirely.
- **The combination traded return for tail safety**: best CVaR and best
  drawdown, worst return. Pick the profile that matches *your* objective.
""")

st.header('7. Limitations')
st.markdown("""
- Only 21 out-of-sample trading days — a single market regime.
- Expected returns are historical means (no predictive model; the original
  project has no forecasting capability).
- No transaction costs, taxes or turnover modelling; static single-shot weights.
- One dominant technology cluster limits true diversification inside this universe.
- Results are estimates under disclosed conventions (252-day annualisation,
  log returns, 5.25% risk-free), not guarantees.
""")

st.header('8. Conclusions (from the report)')
st.markdown("""
1. **Always apply Ledoit-Wolf denoising** — zero cost, no tuning, reduces
   hidden estimation risk even when weights barely move.
2. **Use clustering for risk control**, not return chasing — and only when the
   cluster structure is economically meaningful.
3. **Judge combinations over 12–24 months and multiple regimes**, not one month.
""")

st.header('9. References (MLA excerpts from the report)')
st.markdown("""
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for
  large-dimensional covariance matrices. *Journal of Multivariate Analysis*,
  88(2), 365–411.
- Markowitz, H. (1952). Portfolio selection. *The Journal of Finance*, 7(1), 77–91.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive
  diversification. *The Review of Financial Studies*, 22(5), 1915–1953.
- Michaud, R. O. (1989). The Markowitz optimization enigma. *Financial
  Analysts Journal*, 45(1), 31–42.
- Ao, M., Li, Y., & Zheng, X. (2019). Approaching mean-variance efficiency for
  large portfolios. *The Review of Financial Studies*, 32(7), 2890–2919.
- Park, J. (2022). Clustering approaches for global minimum variance portfolio.
  *Mathematics*, 10(4), 539.
- Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for
  some sets of random matrices. *Matematicheskii Sbornik*, 72(4), 507–536.
""")

disclaimer_box()
