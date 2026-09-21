"""Competition metric: Pearson across proteins, averaged across cells."""

import numpy as np
import torch


def row_normalize(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    z = x - x.mean(axis=1, keepdims=True)
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), eps)


def pearson_rows(y, p):
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] < 2:
        raise ValueError("Expected aligned cells x proteins matrices")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Non-finite input to correlation metric")
    a, b = y - y.mean(1, keepdims=True), p - p.mean(1, keepdims=True)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.clip(
        np.divide((a * b).sum(1), den, out=np.full(len(y), -1.0), where=den > 0), -1, 1
    )


def correlation_loss(p, y):
    a, b = p - p.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
    return 1 - torch.nn.functional.cosine_similarity(a, b, dim=1, eps=1e-8).mean()


def protein_correlations(y, p):
    # Unlike the competition cell-wise penalty, a constant protein is undefined.
    result = pearson_rows(y.T, p.T)
    constant = (np.ptp(y, axis=0) == 0) | (np.ptp(p, axis=0) == 0)
    result[constant] = np.nan
    return result


def paired_cluster_bootstrap(y, a, b, groups, repeats=2000, seed=2026):
    """Resample donor-day clusters, preserving paired predictions; descriptive CI."""
    delta = pearson_rows(y, a) - pearson_rows(y, b)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    sums = np.array([delta[groups == g].sum() for g in unique])
    sizes = np.array([(groups == g).sum() for g in unique])
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(unique), (repeats, len(unique)))
    values = sums[ix].sum(1) / sizes[ix].sum(1)
    return dict(
        delta=float(delta.mean()),
        low=float(np.quantile(values, 0.025)),
        high=float(np.quantile(values, 0.975)),
        clusters=len(unique),
        repeats=repeats,
    )
