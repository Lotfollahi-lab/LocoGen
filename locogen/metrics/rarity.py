"""Rarity score (Han et al., 2023).

For each generated sample that falls inside at least one training k-NN ball
(on-manifold), the rarity is the radius of the smallest such ball. Off-manifold
samples are excluded so the statistic isolates on-manifold sparsity. Higher
means the sample sits in a less dense region of the training distribution.
"""
from typing import Dict

import numpy as np


def compute_rarity_score(real_feats: np.ndarray, fake_feats: np.ndarray,
                         real_kth: np.ndarray) -> Dict[str, float]:
    from scipy.spatial.distance import cdist
    d = cdist(real_feats, fake_feats)                    # (n_real, n_fake)
    contains = d < real_kth[:, None]                     # broadcast
    on_manifold = contains.any(axis=0)                   # (n_fake,)
    radii = np.where(contains, real_kth[:, None], np.inf)
    rarity = radii.min(axis=0)                           # ∞ for off-manifold
    valid = rarity[on_manifold]
    if valid.size == 0:
        nan = float("nan")
        return {"rarity_median": nan, "rarity_mean": nan,
                "rarity_p25": nan, "rarity_p75": nan,
                "on_manifold_frac": 0.0}
    return {
        "rarity_median":    float(np.median(valid)),
        "rarity_mean":      float(valid.mean()),
        "rarity_p25":       float(np.percentile(valid, 25)),
        "rarity_p75":       float(np.percentile(valid, 75)),
        "on_manifold_frac": float(on_manifold.mean()),
    }
