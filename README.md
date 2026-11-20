# 📈 Quant Portfolio Intelligence

**Machine Learning | Quantitative Portfolio Optimization | Investment Risk Analytics**

An interactive, deployable portfolio-management platform built directly on the completed
**MScFE 652 Portfolio Management — Group Work Project 3** research model
(WorldQuant University, individual submission, February 2026).

> **Understand Markets. Optimize Portfolios. Manage Risk.**

The academic model is preserved exactly — nothing replaced, nothing invented:

| Original GWP3 component | Status in this app |
|---|---|
| 12 large-cap U.S. equities (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AVGO, COST, ASML, PEP, NFLX) | ✅ identical tickers, identical column order |
| Training Jan 1 2024 – Jun 30 2025 · OOS test Jul 1–31 2025 | ✅ Environment A preserved |
| Baseline max-Sharpe MVO, long-only, 15% cap (PyPortfolioOpt) | ✅ same objective, bounds `(0, 0.15)` |
| Ledoit-Wolf denoising (α* ≈ 0.0442; cond. 55.8 → 39.5) | ✅ reproduced live |
| K-Medoids clustering (K = 4, correlation distance, seed 42) | ✅ same pure-NumPy implementation |
| Combination pipeline: Denoise → Cluster → MVO | ✅ same 4-step construction |
| Rolling walk-forward backtest (W = 126, S = 21) | ✅ training data only |
| Metric conventions (252-day annualisation, rf = 5.25%) | ✅ identical formulas |

The submitted report's published figures (Sharpe 1.572 train / 0.582 OOS, Combo OOS
−7.29% annualised with the best tail risk, etc.) are clearly labelled as static
report references and are never mixed with freshly computed values.

---

## Features

- **9 professional pages** — Home intelligence, Live Market Dashboard, Portfolio
  Optimization Lab, Denoising & Clustering, Backtesting & Performance, Risk Analytics,
  Investment Insights, Methodology, About the Developer.
- **Two separated data environments** — Environment A (academic GWP3 validation,
  fixed windows) and Environment B (live monitoring with the latest available data).
- **Live data integration** — Yahoo Finance via `yfinance`, adjusted prices, TTL
  caching, freshness timestamps, source attribution, clean error states.
  Data is delayed/end-of-day and is never described as real-time.
- **Dynamic recommendations engine** — every recommendation is generated from actual
  computed outputs (weights, HHI, risk contributions, cluster sizes, α*) with
  investor / retail / company audience views. Nothing hardcoded.
- **Plain-language explanations** — every metric has definition, why-it-matters,
  how-to-read, limitations; full glossary; tooltips everywhere.
- **Tested financials** — pytest suite covering constraint validation, Sharpe/MDD
  hand-checks, LW shrinkage algebra, K-Medoids behaviour, backtest leakage checks.
- **Security review** — see [SECURITY.md](SECURITY.md).

## Technology stack

Python 3.10–3.12 · Streamlit · pandas/numpy/scipy · scikit-learn (LedoitWolf)
· PyPortfolioOpt (EfficientFrontier) · yfinance · Plotly · pytest

## Repository structure

```text
GWP3-Quant-Portfolio-Intelligence/
├── app.py                      # Page 0 — Home / Investment Intelligence
├── requirements.txt            # pinned dependencies
├── README.md · LICENSE · SECURITY.md · .gitignore · .env.example
├── .streamlit/config.toml      # theme, XSRF, upload limits
├── pages/
│   ├── 1_Live_Market_Dashboard.py
│   ├── 2_Portfolio_Optimization.py
│   ├── 3_Denoising_Clustering.py
│   ├── 4_Backtesting_Performance.py
│   ├── 5_Risk_Analytics.py
│   ├── 6_Investment_Insights.py
│   ├── 7_Methodology_Report.py
│   └── 8_About_Developer.py
├── src/
│   ├── config.py               # all model parameters (from the GWP3 notebook)
│   ├── data_loader.py          # yfinance retrieval + TTL cache
│   ├── data_validation.py      # cleaning, returns, integrity checks
│   ├── optimization.py         # max-Sharpe MVO, frontier, combination portfolio
│   ├── denoising.py            # Ledoit-Wolf shrinkage + diagnostics
│   ├── clustering.py           # K-Medoids + risk parity
│   ├── backtesting.py          # rolling walk-forward (W=126, S=21)
│   ├── risk_metrics.py         # the 7 GWP3 metrics, VaR/ES, risk contributions
│   ├── forecasting.py          # clearly-labelled estimates (no fake predictions)
│   ├── recommendations.py      # dynamic insights engine
│   ├── visualizations.py       # Plotly chart library
│   ├── security.py             # input validation, safe errors
│   └── ui_utils.py             # Streamlit helpers, glossary
├── tests/test_all.py           # financial correctness tests
├── notebooks/                  # original GWP3 notebook (submission package)
├── reports/
│   ├── DEPLOYMENT_GUIDE.md     # beginner-friendly GitHub + Streamlit Cloud guide
│   └── USER_GUIDE.md           # visitor guide
└── assets/images/
```

## Quick start (local)

```bash
# 1. clone
git clone https://github.com/ajeetk095/GWP3-Quant-Portfolio-Intelligence.git
cd GWP3-Quant-Portfolio-Intelligence

# 2. create an environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. install pinned dependencies
pip install -r requirements.txt

# 4. run the financial tests (optional but recommended)
python -m pytest tests/ -v

# 5. launch
streamlit run app.py
```

The app opens on the Home page. The full model runs on first load (downloads
Jan 2024 → Jul 2025 daily data for 12 tickers, ~5–20 s) and is then cached.

## Configuration

- No API keys required. Optional future keys go in `.env` (copy `.env.example`) —
  never commit it.
- Theme/XSRF settings: `.streamlit/config.toml`.
- All model parameters live in `src/config.py`; the academic values must not be
  changed if you need to reproduce the GWP3 results.

## Data sources

| Source | Use | Freshness |
|---|---|---|
| Yahoo Finance (yfinance) | daily adjusted prices, latest quotes | delayed / end-of-day |
| Model estimates | expected returns, covariance | computed on refresh (TTL cache 1 h) |

No market observation is ever fabricated: if a source fails, the app shows a
clear error instead of stale or invented numbers.

## Testing

```bash
python -m pytest tests/ -v        # constraints, metrics, shrinkage, clustering, leakage
pip-audit -r requirements.txt     # dependency vulnerabilities
bandit -r src/ app.py pages/      # static security scan
```

## Deployment

See **[reports/DEPLOYMENT_GUIDE.md](reports/DEPLOYMENT_GUIDE.md)** — a
beginner-friendly, step-by-step guide to GitHub + Streamlit Community Cloud
(free, no local server needed).

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Market data could not be loaded" | Yahoo Finance temporarily unreachable — press 🔄 Refresh; check your connection. |
| `PyPortfolioOpt` install error | Ensure Python 3.10–3.12; the app falls back to a SciPy solver with identical constraints if the import fails. |
| Slow first load | First run downloads 18 months of daily data for 12 tickers; afterwards cached ~1 h. |
| Numbers differ slightly from the report | Expected — the report used data downloaded Jan–Jul 2025; the app recomputes with current data and always discloses which is which. |

## Disclaimer

This platform is an educational and quantitative research tool. Outputs are
estimates subject to data limitations and model assumptions — not guaranteed
returns and not personalized financial advice. See the in-app disclaimer.

## Developer

**Ajeet** — *Bridging code & capital.*
GitHub: <https://github.com/ajeetk095> ·
LinkedIn: <https://www.linkedin.com/in/ajeek095/> ·
Email: ajeetk095@gmail.com
