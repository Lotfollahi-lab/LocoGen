"""Motif-usage statistics behind the compositional plausibility axis.

Both are read off a single JASPAR scan: `gini` measures how concentrated motif
usage is across the repertoire, and `hits_per_seq_stats` measures motif
density per sequence. The paper reports each as an absolute deviation from the
training set.
"""
import numpy as np


def gini(x):
    """Gini coefficient of nonneg array; 0 = uniform, 1 = total concentration."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def shannon_entropy(p):
    """Normalized Shannon entropy in [0, 1]; 1 = uniform over support."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    h = -(p * np.log(p)).sum()
    h_max = np.log(len(p))
    return float(h / h_max) if h_max > 0 else 0.0


def hits_per_seq_stats(hits_per_seq) -> dict:
    """Motif density: above-threshold motif occurrences per sequence."""
    h = np.asarray(hits_per_seq, dtype=np.float64)
    return {
        "motif_hits_per_seq_mean": float(h.mean()),
        "motif_hits_per_seq_median": float(np.median(h)),
    }
