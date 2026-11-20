"""
Plotly visualization helpers — shared across all Streamlit pages.
Consistent fintech-style colour palette across the whole app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

METHOD_COLORS = {
    'Baseline (MVO)': '#2196F3',
    'Denoise (LW)': '#FF9800',
    'Cluster (K-Med)': '#4CAF50',
    'Combo (Den+Clust)': '#9C27B0',
}

APP_TEMPLATE = 'plotly_white'


def _fmt_layout(fig: go.Figure, title: str, x: str, y: str) -> go.Figure:
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y,
                      template=APP_TEMPLATE, hovermode='closest',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02))
    return fig


def price_chart(prices: pd.DataFrame, tickers: list[str]) -> go.Figure:
    fig = go.Figure()
    for t in tickers:
        fig.add_trace(go.Scatter(x=prices.index, y=prices[t], name=t,
                                 mode='lines', line=dict(width=1.6)))
    return _fmt_layout(fig, 'Historical Adjusted Close Prices', 'Date', 'Price (USD)')


def normalized_price_chart(prices: pd.DataFrame, tickers: list[str]) -> go.Figure:
    norm = prices[tickers] / prices[tickers].iloc[0] * 100
    fig = go.Figure()
    for t in tickers:
        fig.add_trace(go.Scatter(x=norm.index, y=norm[t], name=t, mode='lines',
                                 line=dict(width=1.4)))
    return _fmt_layout(fig, 'Normalized Performance (start = 100)', 'Date', 'Index')


def allocation_pie(weights: dict[str, float], title: str) -> go.Figure:
    items = {t: w for t, w in weights.items() if w > 0.005}
    fig = go.Figure(go.Pie(labels=list(items), values=list(items),
                           hole=0.42, textinfo='label+percent',
                           textposition='outside', automargin=True))
    fig.update_layout(title=title, template=APP_TEMPLATE, showlegend=False)
    return fig


def weights_bar_chart(weights_by_method: dict[str, list[float]],
                      tickers: list[str]) -> go.Figure:
    fig = go.Figure()
    x = np.arange(len(tickers))
    n = len(weights_by_method)
    for i, (name, w) in enumerate(weights_by_method.items()):
        fig.add_trace(go.Bar(
            x=[t + f'<span style="opacity:0">{" " * i}</span>' for t in tickers]
            if n > 4 else tickers,
            y=np.asarray(w) * 100, name=name,
            marker_color=METHOD_COLORS.get(name, '#888'),
            width=0.8 / n))
    fig.add_hline(y=15, line_dash='dash', line_color='red',
                  annotation_text='15% cap')
    return _fmt_layout(fig, 'Portfolio Weights vs. the 15% Cap', 'Asset', 'Weight (%)')


def risk_return_scatter(mu: pd.Series, S: pd.DataFrame,
                        weights_by_method: dict[str, list[float]]) -> go.Figure:
    fig = go.Figure()
    vols = np.sqrt(np.diag(S.values))
    fig.add_trace(go.Scatter(x=vols, y=mu.values, mode='markers+text',
                             text=list(mu.index), textposition='top center',
                             name='Assets', marker=dict(size=9, color='#555')))
    for name, w in weights_by_method.items():
        w = np.asarray(w)
        pv = float(np.sqrt(w @ S.values @ w))
        pr = float(w @ mu.values)
        fig.add_trace(go.Scatter(x=[pv], y=[pr], mode='markers',
                                 name=name, marker=dict(
                                     size=14, symbol='star',
                                     color=METHOD_COLORS.get(name, '#888'))))
    return _fmt_layout(fig, 'Risk-Return Map (training-period estimates)',
                       'Annualised Volatility', 'Annualised Expected Return')


def frontier_chart(frontier: pd.DataFrame,
                   weights_by_method: dict[str, list[float]],
                   mu: pd.Series, S: pd.DataFrame) -> go.Figure:
    fig = risk_return_scatter(mu, S, weights_by_method)
    if frontier is not None and len(frontier):
        fig.add_trace(go.Scatter(x=frontier['vol'], y=frontier['ret'],
                                 mode='lines', name='Efficient Frontier',
                                 line=dict(color='#1e88e5', width=2.5)))
    return fig


def cumulative_return_chart(cum_by_method: dict[str, pd.Series],
                            title: str) -> go.Figure:
    fig = go.Figure()
    for name, cum in cum_by_method.items():
        fig.add_trace(go.Scatter(x=cum.index, y=(cum - 1) * 100, name=name,
                                 mode='lines',
                                 line=dict(color=METHOD_COLORS.get(name, '#888'),
                                           width=2)))
    fig.add_hline(y=0, line_dash='dot', line_color='black')
    return _fmt_layout(fig, title, 'Date', 'Cumulative Return (%)')


def drawdown_chart(dd_by_method: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for name, dd in dd_by_method.items():
        fig.add_trace(go.Scatter(x=dd.index, y=dd * 100, name=name, mode='lines',
                                 line=dict(color=METHOD_COLORS.get(name, '#888'),
                                           width=1.5),
                                 fill='tozeroy', fillcolor='rgba(150,150,150,0.10)'))
    return _fmt_layout(fig, 'Drawdown Comparison', 'Date', 'Drawdown (%)')


def correlation_heatmap(corr: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=corr.values, x=list(corr.columns),
                               y=list(corr.index), zmin=-1, zmax=1,
                               colorscale='RdYlGn', colorbar=dict(title='ρ'),
                               hovertemplate='%{y} vs %{x}: %{z:.2f}<extra></extra>'))
    fig.update_layout(title=title, template=APP_TEMPLATE)
    return fig


def eigenvalue_chart(eig_sample: np.ndarray, eig_lw: np.ndarray) -> go.Figure:
    idx = np.arange(len(eig_sample))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=idx - 0.2, y=eig_sample, width=0.38,
                         name='Sample covariance', marker_color='#1e88e5'))
    fig.add_trace(go.Bar(x=idx + 0.2, y=eig_lw, width=0.38,
                         name='Ledoit-Wolf shrunk', marker_color='#fb8c00'))
    return _fmt_layout(fig, 'Eigenvalue Spectrum — Shrinkage Compresses Extremes',
                       'Eigenvalue index (sorted descending)', 'Eigenvalue (annualised)')


def cluster_heatmap(corr: pd.DataFrame, labels: np.ndarray,
                    tickers: list[str]) -> go.Figure:
    order = np.argsort(labels)
    ordered = corr.iloc[order, order]
    names = [tickers[i] for i in order]
    fig = correlation_heatmap(ordered, 'Correlation Matrix — Reordered by Cluster')
    fig.update_layout(xaxis=dict(tickvals=list(range(len(names))), ticktext=names),
                      yaxis=dict(tickvals=list(range(len(names))), ticktext=names))
    return fig


def metrics_bar_chart(table: pd.DataFrame, metric: str, period: str) -> go.Figure:
    methods = list(table.index)
    colors = [METHOD_COLORS.get(m, '#888') for m in methods]
    fig = go.Figure(go.Bar(x=methods, y=table[metric], marker_color=colors))
    return _fmt_layout(fig, f'{metric} — {period}', 'Method', metric)


def rolling_sharpe_chart(bt_windows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(bt_windows))),
                             y=bt_windows['Sharpe'], mode='lines+markers',
                             name='OOS Sharpe per window',
                             line=dict(color='#1e88e5')))
    mean_s = bt_windows['Sharpe'].mean()
    fig.add_hline(y=mean_s, line_dash='dash', line_color='red',
                  annotation_text=f'Mean = {mean_s:.3f}')
    return _fmt_layout(fig, 'Walk-Forward Backtest — OOS Sharpe per Window',
                       'Window index', 'Sharpe ratio')


def backtest_wealth_chart(cum_oos: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_oos.index, y=cum_oos.values,
                             mode='lines', name='Wealth index',
                             line=dict(color='#2e7d32', width=2.2)))
    fig.add_hline(y=1.0, line_dash='dot', line_color='black')
    return _fmt_layout(fig, 'Cumulative OOS Wealth — Rolling Denoised Backtest',
                       'Date', 'Wealth (starts at 1)')


def risk_contribution_chart(rc: pd.Series, weights: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(rc.index), y=rc.values * 100,
                         name='Risk contribution (%)',
                         marker_color='#e53935'))
    fig.add_trace(go.Bar(x=list(weights.index), y=weights.values * 100,
                         name='Weight (%)', marker_color='#1e88e5'))
    return _fmt_layout(fig, 'Weight vs. Risk Contribution (Baseline)',
                       'Asset', 'Percent')


def cluster_allocation_chart(cluster_alloc: dict) -> go.Figure:
    fig = go.Figure(go.Pie(labels=list(cluster_alloc), values=list(cluster_alloc.values),
                           hole=0.42, textinfo='label+percent'))
    fig.update_layout(title='Combination Portfolio — Cluster-Level Allocation',
                      template=APP_TEMPLATE, showlegend=False)
    return fig
