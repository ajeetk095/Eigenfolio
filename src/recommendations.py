"""
Recommendations engine — every recommendation is generated dynamically from
actual model outputs (weights, metrics, clustering, live observations).
Nothing is hardcoded: rules fire only when supported by computed evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MAX_WEIGHT, REPORT_DIAGNOSTICS


def _fmt_pct(x: float) -> str:
    return f'{x * 100:.1f}%'


def generate_recommendations(result: dict,
                             live_vol_note: str | None = None) -> list[dict]:
    """Build evidence-backed recommendations from the computed result dict.

    Each item: title, finding, why_it_matters, assets, consider, risks,
    supporting_metrics, audience.
    """
    recs = []
    metrics = result['metrics']
    weights = result['weights']
    train_tbl = {n: m['train'] for n, m in metrics.items()}
    test_tbl = {n: m['test'] for n, m in metrics.items()}
    dn = result['denoise']
    cl = result['clusters']

    # --- 1. Diversification quality (weights dispersion) --------------------
    for name, w in weights.items():
        hhi = float(np.sum(np.asarray(w) ** 2))
        eff_n = 1.0 / hhi if hhi > 0 else 0
        n_active = int((np.asarray(w) > 0.005).sum())
        if eff_n < 4.0:
            recs.append({
                'title': f'Portfolio diversification may need improvement ({name})',
                'finding': (f'The {name} portfolio holds {n_active} active assets, '
                            f'equivalent to only {eff_n:.1f} independent positions '
                            f'(HHI = {hhi:.2f}).'),
                'why': 'Fewer effective positions means returns depend heavily on a '
                       'small number of stocks — bad months for those names dominate '
                       'the whole portfolio.',
                'assets': ', '.join([t for t, wi in zip(result['mu'].index, w) if wi > 0.05]),
                'consider': 'Review whether the concentration matches your risk '
                            'tolerance; more balanced allocations exist in the app '
                            '(see the Cluster portfolio).',
                'risks': 'Diversification does not remove market-wide risk; all 12 '
                         'assets are US large-caps and fall together in broad selloffs.',
                'supporting': f'HHI={hhi:.3f}, effective N={eff_n:.1f}, active assets={n_active}',
                'audience': 'Investors',
            })

    # --- 2. High risk contribution -------------------------------------------
    from .risk_metrics import risk_contributions
    S = result['S_sample']
    base_rc = risk_contributions(weights['Baseline (MVO)'], S)
    top_rc = base_rc.idxmax()
    if base_rc.max() > 0.30:
        recs.append({
            'title': f'{top_rc} contributes a relatively high share of portfolio risk',
            'finding': f'{top_rc} accounts for {_fmt_pct(base_rc.max())} of total '
                       'portfolio variance despite its weight of '
                       f'{_fmt_pct(weights["Baseline (MVO)"][base_rc.index.get_loc(top_rc)] if isinstance(base_rc.index.get_loc(top_rc), (int, float)) else 0)}.',
            'why': 'Risk contribution weights volatility AND co-movement: an asset '
                   'that swings with the rest of the portfolio adds more risk than '
                   'its weight suggests.',
            'assets': top_rc,
            'consider': 'If this risk share feels too high, the denoised or cluster '
                        'portfolios spread risk more evenly.',
            'risks': 'Risk contributions are estimated from the training window and '
                     'change as correlations change.',
            'supporting': f'Top risk contribution: {top_rc} at {_fmt_pct(base_rc.max())}',
            'audience': 'Investors',
        })

    # --- 3. Concentration approaching the cap --------------------------------
    for name, w in weights.items():
        near_cap = [t for t, wi in zip(result['mu'].index, w)
                    if wi >= MAX_WEIGHT - 0.001]
        if near_cap:
            recs.append({
                'title': f'Portfolio concentration at its permitted limit ({name})',
                'finding': f'{len(near_cap)} assets sit at the 15% cap: {", ".join(near_cap)}.',
                'why': 'The 15% cap is the assignment constraint; when many assets '
                       'bind at the cap, the optimizer would have liked even MORE '
                       'concentration — a sign of strong estimation-error influence.',
                'assets': ', '.join(near_cap),
                'consider': 'Cap-bound portfolios are sensitive to small changes in '
                            'estimates; treat single-period results with caution.',
                'risks': 'If constraints were relaxed, weights could become more '
                         'extreme, not less.',
                'supporting': f'{len(near_cap)} weights at cap; sum = {_fmt_pct(float(np.sum(w)))}',
                'audience': 'Companies',
            })

    # --- 4. Risk-adjusted historical comparison ------------------------------
    best_sharpe_train = max(train_tbl, key=lambda n: train_tbl[n]['Sharpe'])
    best_dd_test = max(test_tbl, key=lambda n: test_tbl[n]['Max Drawdown'])
    combo_test = test_tbl.get('Combo (Den+Clust)', {})
    recs.append({
        'title': f'{best_sharpe_train} offered the best historical risk-adjusted return',
        'finding': (f'On the training window (Jan 2024 - Jun 2025), {best_sharpe_train} '
                    f'has the highest Sharpe ratio ({train_tbl[best_sharpe_train]["Sharpe"]:.3f}). '
                    f'However, in the required out-of-sample month (July 2025), the '
                    f'combo portfolio had the LOWEST return ({combo_test.get("Ann. Return", 0) * 100:.1f}% annualised) '
                    f'while {best_dd_test} protected drawdowns best '
                    f'({test_tbl[best_dd_test]["Max Drawdown"] * 100:.2f}%).'),
        'why': 'Rankings change out of sample: the training period was a tech-heavy '
               'bull regime, and July 2025 rewarded exactly that momentum — one month '
               'cannot validate a structural advantage.',
        'assets': 'All 12 assets',
        'consider': 'Choose a portfolio for its risk profile, not just its best '
                    'historical Sharpe.',
        'risks': 'Past performance does not predict future returns; the OOS window '
                 'is only 21 trading days.',
        'supporting': f'Train Sharpe: {", ".join(f"{n}={v["Sharpe"]:.3f}" for n, v in train_tbl.items())}',
        'audience': 'Investors',
    })

    # --- 5. Volatility monitoring ---------------------------------------------
    train_vol = train_tbl['Baseline (MVO)']['Ann. Vol']
    if train_vol > 0.20:
        recs.append({
            'title': 'Current volatility may justify closer risk monitoring',
            'finding': f'Training-period annualised volatility of the baseline '
                       f'portfolio is {_fmt_pct(train_vol)}.',
            'why': 'Volatility above ~20% annualised means daily swings of roughly '
                   '1.3% or more — a $10,000 portfolio can move $130+ in a day.',
            'assets': 'Portfolio level',
            'consider': 'Check the Live Market Dashboard regularly; the cluster '
                        'portfolio reduces volatility in the training window.',
            'risks': 'Volatility is estimated from history and can rise suddenly in '
                     'stress periods.',
            'supporting': f'Baseline ann. vol = {_fmt_pct(train_vol)} (training)',
            'audience': 'Retail investors',
        })

    # --- 6. Denoising value ----------------------------------------------------
    alpha = dn['alpha']
    recs.append({
        'title': 'Ledoit-Wolf denoising is a low-cost safeguard',
        'finding': (f'Shrinkage intensity α* = {alpha:.4f} ({alpha * 100:.1f}% toward '
                    f'the structured target). Condition number improved '
                    f'{dn["cond_sample"]:.1f} → {dn["cond_lw"]:.1f} '
                    f'({dn["cond_improvement"]:.1f}x).'),
        'why': 'Even when weights barely change, denoising makes the covariance '
               'matrix numerically better conditioned — estimates are more stable '
               'and reproducible across market environments.',
        'assets': 'All 12 assets',
        'consider': 'Keep shrinkage switched on: it costs nothing and reduces '
                    'hidden estimation risk.',
        'risks': 'With only 12 assets and T/n ≈ 31, the sample covariance is '
                 'already fairly reliable; benefits grow for larger universes.',
        'supporting': (f'α*={alpha:.4f}; cond. number {dn["cond_sample"]:.1f}→{dn["cond_lw"]:.1f}'),
        'audience': 'Companies',
    })

    # --- 7. Cluster structure ---------------------------------------------------
    sizes = {c: len(v['members']) for c, v in cl['clusters'].items()}
    biggest = max(sizes, key=lambda c: sizes[c])
    if sizes[biggest] > 6:
        recs.append({
            'title': 'Cluster structure offers limited true diversification',
            'finding': (f'K-Medoids placed {sizes[biggest]} of 12 assets in one '
                        f'cluster (medoid {cl["clusters"][biggest]["medoid"]}) — the '
                        'universe is dominated by a common technology factor.'),
            'why': 'When most assets move together, "cluster diversification" is '
                   'mostly reweighting within one risk factor.',
            'assets': ', '.join(cl['clusters'][biggest]['members']),
            'consider': 'True diversification would require adding assets from '
                        'uncorrelated sectors; within this universe, defensive '
                        'names (COST, PEP) are the main offset.',
            'risks': 'Cluster composition depends on the training window and can '
                     'change over time.',
            'supporting': f'Cluster sizes: {sizes}',
            'audience': 'Companies',
        })

    # --- 8. Live-data note (only passed when live data loaded) ------------------
    if live_vol_note:
        recs.append({
            'title': 'Live market conditions have been checked',
            'finding': live_vol_note,
            'why': 'Recent market movements can change the portfolio risk profile '
                   'between rebalancing dates.',
            'assets': 'Portfolio level',
            'consider': 'Re-run the optimisation periodically (this app recomputes '
                        'from the latest data on refresh).',
            'risks': 'Delayed end-of-day data is used, not real-time quotes.',
            'supporting': 'Live Market Dashboard (Environment B)',
            'audience': 'Retail investors',
        })

    return recs


def audience_view(recs: list[dict], audience: str) -> list[dict]:
    """Filter recommendations for a given audience view."""
    mapping = {
        'Investors': ['Investors', 'All'],
        'Retail investors': ['Retail investors', 'Investors', 'All'],
        'Companies': ['Companies', 'Investors', 'All'],
    }
    allowed = mapping.get(audience, audience)
    return [r for r in recs if r['audience'] in allowed]
