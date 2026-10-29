"""Authenticity (Alaa et al., 2022).

Fraction of generated samples that lie farther from their nearest training
neighbour than that neighbour's own leave-one-out 1-NN distance. Higher means
less memorisation of the training set.
"""
from typing import Dict

import numpy as np


def compute_authenticity(real_feats: np.ndarray, fake_feats: np.ndarray,
                         train_nn1: np.ndarray) -> Dict[str, float]:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(real_feats)
    d_g, idx = nn.kneighbors(fake_feats)
    authentic = d_g.ravel() > train_nn1[idx.ravel()]
    return {"authenticity": float(authentic.mean())}
