# Project instructions and reproducibility guide

> This is a project-authored guide to running and understanding this repository. It is **not** the official WorldQuant University assignment handout. The official handout was not included in the repository; add it only if you have permission to redistribute it.

## Objective

Reproduce the MScFE 652 GWP3 portfolio study and explore its published model through a Streamlit application. The original assignment evaluation must remain distinct from live-market context.

## Reproduction windows and rules

- Training prices: 2024-01-01 through 2025-06-30.
- Required out-of-sample test: 2025-07-01 through 2025-07-31.
- Universe: the 12 symbols configured in `src/config.py`.
- Constraints: fully invested, long-only, and maximum 15% per asset.
- Annualization: 252 trading days; the report convention uses a 5.25% annual risk-free rate.

## Methods represented

1. Baseline maximum-Sharpe mean-variance optimization.
2. Ledoit–Wolf covariance shrinkage.
3. Correlation-distance K-Medoids clustering.
4. A denoise/cluster/optimization combination.
5. A separate rolling walk-forward research backtest.

## Running locally

1. Use a supported Python version and install the repository dependencies with `pip install -r requirements.txt`.
2. Run the tests with `python -m pytest tests/ -v`.
3. Start the interface with `streamlit run app.py`.
4. Market data is fetched from Yahoo Finance and may be delayed, rate-limited, or unavailable. Results are not fabricated when data is missing.

## Interpretation

Live history is a monitoring/context feature and does not change the fixed academic train/test split. Historical estimates are not forecasts, investment advice, or guarantees of return. The project does not execute trades or model all transaction costs.
