"""Per-sequence k-mer feature space.

The novelty metrics compare generated and training sequences in the space of
per-sequence k-mer frequency vectors; `kth_nn_distance` supplies the training
k-NN radii that both Authenticity and Rarity are measured against.
"""
import numpy as np

from locogen.utils.seq_stats import kmer_counts


def per_seq_kmer_freq(seqs, k):
    """(N, 4^k) per-sequence k-mer frequency matrix (each row sums to 1).

    Used as the canonical-MMD input — one observation per sequence, in a
    model-free representation. RBF kernel is well-defined here (low-dim
    continuous, bandwidth via median heuristic)."""
    from locogen.utils.seq_stats import kmer_counts
    out = np.zeros((len(seqs), 4 ** k), dtype=np.float32)
    for i, s in enumerate(seqs):
        c = kmer_counts(s, k).astype(np.float32)
        denom = c.sum()
        if denom > 0:
            out[i] = c / denom
    return out


def kth_nn_distance(X: np.ndarray, k: int) -> np.ndarray:
    """Distance from each row of X to its k-th nearest neighbour in X
    (excluding self). Shape: (N,).
    """
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    dists, _ = nbrs.kneighbors(X)
    return dists[:, k]
