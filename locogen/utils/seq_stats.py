"""Sequence statistics shared by the evaluation scripts.

`kmer_counts`, `group_kmer_distribution` and `jensen_shannon` are lifted
verbatim from the separator-analysis module of the authors' research tree so
that the published evaluation path does not depend on that analysis entry
point. They back the k-mer Jensen-Shannon novelty metrics computed by
`scripts.eval.compute_novelty_extended`.
"""
from typing import List

import numpy as np


def kmer_counts(s: str, k: int) -> np.ndarray:
    """Counts of all 4**k k-mers (alphabet ACGT) in s, in fixed lexicographic order."""
    dim = 4 ** k
    v = np.zeros(dim, dtype=np.int64)
    if len(s) < k:
        return v
    idx_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    cur = 0
    valid = 0
    mask = dim - 1
    for ch in s:
        if ch not in idx_map:
            valid = 0
            cur = 0
            continue
        cur = ((cur << 2) | idx_map[ch]) & mask
        valid += 1
        if valid >= k:
            v[cur] += 1
    return v


def group_kmer_distribution(seqs: List[str], k: int) -> np.ndarray:
    dim = 4 ** k
    total = np.zeros(dim, dtype=np.int64)
    for s in seqs:
        total += kmer_counts(s, k)
    return total.astype(float)


def jensen_shannon(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        return float(np.sum(a * (np.log2(a) - np.log2(b))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
