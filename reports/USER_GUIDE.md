# 👥 Visitor User Guide — Quant Portfolio Intelligence

**Who is this for?** Anyone curious about how a professionally optimized
portfolio works — investors, retail investors, students, and financial
professionals. No quantitative-finance, maths or coding background needed.

## What the app does
It takes a completed academic portfolio-management study (WorldQuant University
GWP3) and turns it into an interactive platform: a portfolio of 12 well-known
U.S. stocks (Apple, Microsoft, Nvidia, Costco, PepsiCo…), the classic
optimization method, three machine-learning improvements, live market data,
and honest performance testing.

## What the app cannot do
- Predict future prices or guarantee any return.
- Give personalized financial advice.
- Show real-time tick-by-tick quotes (data is delayed / end-of-day).
- Trade or connect to any brokerage account.

## Page-by-page

**0. Home** — headline KPIs, current risk assessment, and today's top
recommendations. Start here. Every number shown is computed live from market
data, and the small print tells you exactly when the data was fetched.

**1. Live Market Dashboard** — latest prices, daily changes, charts,
correlations. Use the asset selector and date slider; press 🔄 Refresh for
fresher data (cached ~10 minutes).

**2. Portfolio Optimization** — four portfolio tabs (Baseline, Denoised,
Cluster, Combination) with allocations, constraint checks, expected return,
volatility, Sharpe and Sortino. The 15% cap and long-only rules are always
enforced and verified on screen.

**3. Denoising & Clustering** — the ML machinery: shrinkage intensity α*, the
eigenvalue chart, raw vs. denoised correlations, and the K = 4 cluster map.
Expanders explain each concept in one paragraph.

**4. Backtesting & Performance** — training vs. out-of-sample tables with the
🏆 winner of every metric, cumulative-return and drawdown charts, plus the
rolling walk-forward analysis. Negative results are shown, never hidden.

**5. Risk Analytics** — volatility, drawdown, VaR/ES, risk contributions and a
"drop one asset" sensitivity table. Every metric has a "What does this mean?"
explainer with its limitations.

**6. Investment Insights** — the full recommendations engine. Choose your view
(investor / retail / company); each recommendation states what was found, why
it matters, which assets, what to consider, what risks remain, and which
metrics support it.

**7. Methodology** — the whole academic project in plain language, including
the published report numbers and the honest finding that the ML combination
portfolio *lost* on returns in the July 2025 test.

**8. About** — the developer profile and working contact links.

## How to read a portfolio allocation
A pie or table showing MSFT 10%, NVDA 15%… means: of every $100 invested,
$10 goes to Microsoft, $15 to Nvidia, etc. Weights always sum to 100%, no
asset exceeds 15%, and nothing is ever "shorted".

## How to interpret recommendations
They are **research observations about historical and current estimated
behaviour**, each with explicit risks and uncertainties. They are not trading
signals, and the app never claims certainty about future prices.

## Understanding forecast uncertainty
The only forward-looking numbers are expected-return and volatility *estimates*
derived from history. Each shows its method, horizon, training window and
uncertainty band. Treat them as inputs to thinking, not promises.
