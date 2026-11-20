"""
Step 2/3 ML improvement — Clustering via K-Medoids + inverse-volatility parity.

Preserved from the original GWP3 notebook:
- Correlation distance  d(i,j) = sqrt((1 - rho_ij) / 2)
- Pure-NumPy K-Medoids (k-medoids++ seeding, PAM-style swaps), seed = 42
- Intra-cluster inverse-volatility weights; equal 1/K cluster budgets;
  15% per-asset cap applied, then renormalisation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MAX_WEIGHT, N_CLUSTERS, RANDOM_SEED
from .optimization import enforce_cap_iterative


def kmedoids_fit(dist_matrix: np.ndarray, n_clusters: int = N_CLUSTERS,
                 random_state: int = RANDOM_SEED, max_iter: int = 500):
    """K-Medoids clustering on a precomputed distance matrix.

    Returns (labels, medoid_indices).
    """
    rng = np.random.default_rng(random_state)
    n = dist_matrix.shape[0]

    # k-medoids++ style distance-proportional seeding
    first = int(rng.integers(n))
    medoids = [first]
    for _ in range(n_clusters - 1):
        d_nearest = np.min(dist_matrix[:, medoids], axis=1)
        probs = d_nearest ** 2
        probs = probs / probs.sum() if probs.sum() > 0 else np.full(n, 1 / n)
        medoids.append(int(rng.choice(n, p=probs)))

    medoids = list(dict.fromkeys(medoids))
    while len(medoids) < n_clusters:
        medoids.append(int(rng.integers(n)))

    def assign(meds):
        return np.argmin(dist_matrix[:, meds], axis=1)

    for _ in range(max_iter):
        labels = assign(medoids)
        new_medoids = medoids.copy()
        changed = False
        for k in range(n_clusters):
            cluster_pts = np.where(labels == k)[0]
            if len(cluster_pts) == 0:
                continue
            intra = dist_matrix[np.ix_(cluster_pts, cluster_pts)].sum(axis=1)
            best = cluster_pts[np.argmin(intra)]
            if best != new_medoids[k]:
                new_medoids[k] = best
                changed = True
        medoids = new_medoids
        if not changed:
            break

    labels = assign(medoids)
    return labels, np.array(medoids)


def correlation_distance(corr: pd.DataFrame) -> np.ndarray:
    """d(i,j) = sqrt((1 - rho_ij)/2), clipped to [0, 1]."""
    c = corr.values
    return np.sqrt(np.clip((1 - c) / 2, 0, 1))


def cluster_portfolio(train_returns: pd.DataFrame, tickers: list[str],
                      n_clusters: int = N_CLUSTERS) -> dict:
    """Build the pure cluster (K-Medoids + risk parity) portfolio."""
    corr = train_returns.corr()
    dist = correlation_distance(corr)
    labels, medoid_idx = kmedoids_fit(dist, n_clusters=n_clusters)

    daily_vols = train_returns.std().values
    weights = np.zeros(len(tickers))

    clusters = {}
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        members = [tickers[i] for i in idx]
        medoid = tickers[medoid_idx[c]]
        clusters[c] = {'members': members, 'medoid': medoid, 'idx': idx}

        inv_vol = 1.0 / daily_vols[idx]
        intra_w = inv_vol / inv_vol.sum()
        weights[idx] = intra_w / n_clusters

    weights = enforce_cap_iterative(weights)

    return {'labels': labels, 'medoid_idx': medoid_idx, 'clusters': clusters,
            'weights': weights, 'corr': corr, 'dist': dist}


def cluster_return_series(train_returns: pd.DataFrame, labels: np.ndarray,
                          tickers: list[str], n_clusters: int = N_CLUSTERS):
    """Cluster-level return series using intra-cluster inverse-vol weights."""
    daily_vols = train_returns.std().values
    intra_weights = {}          # cluster c -> intra-cluster weight vector
    series = {}
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        inv_vol = 1.0 / daily_vols[idx]
        intra_w = inv_vol / inv_vol.sum()
        intra_weights[c] = (idx, intra_w)
        series[f'C{c + 1}'] = train_returns.iloc[:, idx].values @ intra_w
    return pd.DataFrame(series), intra_weights
