# 📖 GWP3 Project Explanation — in Simple Language

A plain-language, section-by-section walkthrough of the submitted report
*"ML Improvements to Portfolio Optimization — Denoising via Ledoit-Wolf
Covariance Shrinkage"* (MScFE 652, individual submission, February 2026).

## 1. Introduction & Portfolio Overview
- **Research question:** Can machine learning reduce the estimation error that
  makes classic portfolio optimization unreliable?
- **The problem:** Markowitz optimization needs expected returns (μ) and the
  covariance matrix (Σ), which must be estimated from history. Estimation error
  flows into the weights, producing unstable, over-concentrated portfolios.
- **Dataset:** 12 large-cap U.S. stocks, daily adjusted closes (Yahoo Finance),
  Jan 1 2024 – Jun 30 2025 for training (373 days) and July 2025 for testing
  (21 days). Rules: no shorting, max 15% per stock, risk-free 5.25%.
- **Actual result of this section:** Table 1 of the report lists all 12 tickers
  and sectors — reproduced exactly in this app.

## 2. Step 1 — Baseline Mean-Variance Optimization
- **Method:** maximize the Sharpe ratio subject to the two rules.
- **Actual result:** the optimizer chose 7 assets; 6 sat at the 15% cap
  (NVDA, META, AVGO, COST, PEP, NFLX) plus MSFT at 10%; AAPL, GOOGL, AMZN,
  TSLA, ASML got zero weight. Training Sharpe **1.572** (return 44.23%,
  volatility 24.80%), OOS Sharpe **0.582** — a 63% drop that is the report's
  clearest evidence of overfitting.

## 3. Step 2 — Denoising (the chosen topic)
- **The intuition:** the "how assets move together" table is noisy. Shrinkage
  blends it 95.6% original / 4.4% simple-average (α* = 0.0442, computed
  analytically). The matrix condition number improved 55.8 → 39.5 (1.4×).
- **Actual result:** identical weights to the baseline — with only 4.4%
  shrinkage the 15% cap binds first. The benefit is hidden but real:
  numerically more stable, more reproducible estimates.
- **Limitation:** denoising fixes Σ only; it never fixes a bad μ estimate.

## 4. Step 3 — Applying it & the Combination Portfolio
- **Clustering (cross-check):** K-Medoids with correlation distance put 9 of 12
  stocks in one technology cluster; TSLA, PEP, COST stood alone. The pure
  cluster portfolio is the most balanced but earns much less (train Sharpe
  0.573) because it ignores expected returns.
- **Combination (Denoise → Cluster → MVO):** cluster-level optimization spread
  money across 9–10 stocks (COST 24.14% after cap renormalisation), cutting
  NVDA/META/PEP exposure. Train Sharpe 1.231 with the lowest volatility
  (22.68%) and best CVaR (−53.22%).

## 5. Step 4 — Multi-Metric Comparison
- **Actual training results:** Baseline/Denoise best Sharpe (1.572); Combo best
  volatility and CVaR; pure Cluster weakest returns but most balanced.
- **Actual OOS (July 2025) results:** Baseline/Denoise best return metrics
  (Sharpe 0.582); Cluster best drawdown protection (−1.84%); Combo *worst*
  return (−7.29% annualised, Sharpe −1.318) but best tail risk (CVaR −17.67%,
  max DD −1.65%).
- **Why:** July 2025 rewarded concentrated tech momentum (NVDA, META, NFLX)
  and penalized COST's 24% weight; PEP's exclusion also hurt.

## 6. Step 5 — Optimal ML Portfolio & OOS Analysis
- **Selected:** the Combination portfolio — chosen on ex-ante theoretical
  grounds (cleanest covariance, diversification, T/n ≈ 93 at cluster level),
  **not** on its one-month OOS return.
- **The three required answers:** (1) differences are regime-driven;
  (2) COST overweight + one dominant cluster + PEP exclusion + wrong regime
  explain the underperformance; (3) denoising always justified (free, no
  tuning), clustering conditional, combination justified only over 12–24 month
  multi-regime horizons — a 21-day test cannot validate it.

## 7. Conclusion
- Always apply LW denoising (zero cost, no downside).
- Use clustering for risk control, not return chasing.
- Judge combinations across multiple regimes; one month proves nothing.

*(All figures above are the published report values, preserved verbatim.)*
