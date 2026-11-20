"""
GWP3 Quant Portfolio Intelligence — central configuration.

Every parameter below is taken directly from the original GWP3 notebook
(MScFE_652_GWP3_Complete.ipynb) and the submitted report (PMPR_Report_3.pdf).
Do not change these values if you need to reproduce the academic results.
"""

# ---------------------------------------------------------------- tickers ---
# Column order matters: weight arrays are indexed in exactly this order.
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META',
           'TSLA', 'AVGO', 'COST', 'ASML', 'PEP', 'NFLX']

SECTORS = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Comm. Services',
    'AMZN': 'Consumer Disc.', 'NVDA': 'Semiconductors', 'META': 'Comm. Services',
    'TSLA': 'Consumer Disc.', 'AVGO': 'Semiconductors', 'COST': 'Consumer Staples',
    'ASML': 'Semiconductors', 'PEP': 'Consumer Staples', 'NFLX': 'Comm. Services',
}

ASSET_DESCRIPTIONS = {
    'AAPL': 'Apple Inc. — consumer hardware & software',
    'MSFT': 'Microsoft Corp. — cloud & enterprise software',
    'GOOGL': 'Alphabet Inc. — search, cloud & AI',
    'AMZN': 'Amazon.com — e-commerce & cloud (AWS)',
    'NVDA': 'NVIDIA Corp. — GPUs, AI accelerators',
    'META': 'Meta Platforms — social media & VR',
    'TSLA': 'Tesla Inc. — electric vehicles & energy',
    'AVGO': 'Broadcom Inc. — networking chips & software',
    'COST': 'Costco Wholesale — warehouse retail',
    'ASML': 'ASML Holding — EUV lithography equipment',
    'PEP': 'PepsiCo Inc. — beverages & snacks (defensive)',
    'NFLX': 'Netflix Inc. — streaming & content',
}

# ------------------------------------------------------------- periods ------
TRAIN_START = '2024-01-01'
TRAIN_END = '2025-06-30'      # academic training window (373 trading days)
TEST_START = '2025-07-01'
TEST_END = '2025-07-31'       # required out-of-sample window (21 trading days)

# ------------------------------------------------------------ parameters ----
RF_ANNUAL = 0.0525            # ~Fed Funds rate 2025, per the report
RF_DAILY = RF_ANNUAL / 252
MAX_WEIGHT = 0.15             # assignment rule: no asset above 15%
N_CLUSTERS = 4                # K for K-Medoids
TRADING_DAYS = 252            # annualisation convention (disclosed in report)
BT_WINDOW = 126               # walk-forward in-sample window (~6 months)
BT_STEP = 21                  # walk-forward OOS step (~1 month)
RANDOM_SEED = 42

# ------------------------------------------------- static report reference --
# Figures published in the submitted GWP3 report (PMPR_Report_3.pdf, Feb 2026),
# computed there from real Yahoo Finance data. Shown in the app ONLY as a
# clearly-labelled static reference when live reproduction is unavailable —
# never presented as freshly computed results.
REPORT_RESULTS = {
    'Baseline (MVO)': {
        'train': {'Ann. Return': 0.4423, 'Ann. Vol': 0.2480, 'Sharpe': 1.572,
                  'Sortino': 2.185, 'Max Drawdown': -0.2190, 'CVaR 95%': -0.5782,
                  'Calmar': 2.020},
        'test': {'Ann. Return': 0.1285, 'Ann. Vol': 0.1307, 'Sharpe': 0.582,
                 'Sortino': 0.827, 'Max Drawdown': -0.0213, 'CVaR 95%': -0.2469,
                 'Calmar': 6.029},
    },
    'Denoise (LW)': {
        'train': {'Ann. Return': 0.4423, 'Ann. Vol': 0.2480, 'Sharpe': 1.572,
                  'Sortino': 2.185, 'Max Drawdown': -0.2190, 'CVaR 95%': -0.5782,
                  'Calmar': 2.020},
        'test': {'Ann. Return': 0.1285, 'Ann. Vol': 0.1307, 'Sharpe': 0.582,
                 'Sortino': 0.827, 'Max Drawdown': -0.0213, 'CVaR 95%': -0.2469,
                 'Calmar': 6.029},
    },
    'Cluster (K-Med)': {
        'train': {'Ann. Return': 0.1892, 'Ann. Vol': 0.2385, 'Sharpe': 0.573,
                  'Sortino': 0.824, 'Max Drawdown': -0.2566, 'CVaR 95%': -0.5434,
                  'Calmar': 0.737},
        'test': {'Ann. Return': 0.1007, 'Ann. Vol': 0.1364, 'Sharpe': 0.353,
                 'Sortino': 0.433, 'Max Drawdown': -0.0184, 'CVaR 95%': -0.2900,
                 'Calmar': 5.471},
    },
    'Combo (Den+Clust)': {
        'train': {'Ann. Return': 0.3317, 'Ann. Vol': 0.2268, 'Sharpe': 1.231,
                  'Sortino': 1.682, 'Max Drawdown': -0.2188, 'CVaR 95%': -0.5322,
                  'Calmar': 1.516},
        'test': {'Ann. Return': -0.0729, 'Ann. Vol': 0.0951, 'Sharpe': -1.318,
                 'Sortino': -2.134, 'Max Drawdown': -0.0165, 'CVaR 95%': -0.1767,
                 'Calmar': -4.418},
    },
}

# Key reported diagnostics (report sections 3.1 / 3.3)
REPORT_DIAGNOSTICS = {
    'shrinkage_alpha': 0.0442,
    'cond_number_sample': 55.8,
    'cond_number_lw': 39.5,
    'train_days': 373,
    'test_days': 21,
}

# ------------------------------------------------------- developer profile --
DEVELOPER = {
    'headline': 'Bridging code & capital.',
    'intro': ('Quantitative finance specialist with an M.Sc. in Financial '
              'Engineering and a B.Tech in IT. I turn complex data into alpha '
              'using Python, C++, SQL, and machine learning — from Markowitz '
              'Mean-Variance Optimization and Black-Litterman to Monte Carlo '
              'simulation and deep learning.'),
    'achievements': [
        '1st Place, IIT Kanpur ML Hackathon',
        'Felicitated by the President of India',
        'AI systems built at Algo8.ai and Amul',
    ],
    'career_focus': ('Seeking full-time Quant Research / Financial Engineering '
                     'roles to solve high-stakes problems where finance meets '
                     'engineering.'),
    'skills': ['Quantitative Finance', 'Financial Engineering',
               'Portfolio Optimization', 'Machine Learning', 'Artificial Intelligence',
               'Python', 'C++', 'SQL', 'Financial Econometrics', 'Risk Analytics',
               'Data Science'],
    'github': 'https://github.com/ajeetk095',
    'linkedin': 'https://www.linkedin.com/in/ajeek095/',
    'email': 'ajeetk095@gmail.com',
}

DISCLAIMER = (
    'This platform is an educational and quantitative research tool. Its '
    'calculations and estimates are subject to data limitations, model '
    'assumptions, and market uncertainty. Outputs are not guaranteed '
    'investment returns or personalized financial advice. Users should '
    'independently assess suitability and consult appropriately qualified '
    'professionals where necessary.'
)

DATA_SOURCE_LABEL = 'Yahoo Finance (yfinance) — delayed / end-of-day data, not tick-by-tick real-time.'
