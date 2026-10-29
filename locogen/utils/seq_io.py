"""Sequence loading and encoding helpers shared across the evaluation axes.

The CSV loaders and the tensor/one-hot encoders used to live in the research
tree's `evaluate_sequences` and `measure_activity_cliff` modules; they are
collected here so each axis script depends only on the package.
"""
import csv
import os

import numpy as np
import pandas as pd
import torch

# Run tree that relative `runs/...` paths are resolved against. Defaults to the
# repository root; override with LOCOGEN_PROJECT_ROOT.
PROJECT_ROOT = os.environ.get(
    "LOCOGEN_PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
NUC = ["A", "C", "G", "T"]
NUC_TO_IDX = {c: i for i, c in enumerate(NUC)}


def load_sequences_csv(path: str, seq_col: str = "seq",
                       cell_type: str = None,
                       cell_type_col: str = "cell_type"):
    """Load sequences from CSV, optionally filtering by cell_type value.

    If `cell_type` is set, only rows whose `cell_type_col` matches are kept.
    Matching is case-insensitive substring (so 'HepG2' matches
    'HepG2_ENCLB441ZZZ' from DNA-Diffusion's tag naming).
    """
    seqs = []
    seen_any_match_col = False
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if cell_type is not None:
                if cell_type_col not in row:
                    continue
                seen_any_match_col = True
                if cell_type.lower() not in row[cell_type_col].lower():
                    continue
            seqs.append(row[seq_col].upper())
    if cell_type is not None and not seen_any_match_col:
        raise ValueError(
            f"--cell_type={cell_type!r} requested but column {cell_type_col!r} "
            f"not present in {path}. Drop the flag or use a CSV with that column."
        )
    return seqs


def load_train_sequences(path: str, seq_col: str = "seq", max_seqs: int = None):
    seqs = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_seqs and i >= max_seqs:
                break
            seqs.append(row[seq_col].upper())
    return seqs


def seqs_to_tensor(seqs):
    return torch.tensor(
        [[BASE_TO_INT.get(c, 0) for c in s] for s in seqs], dtype=torch.long
    )


def gc_content(seq: str) -> float:
    gc = sum(1 for c in seq if c in "GC")
    return gc / len(seq) if len(seq) > 0 else 0.0


SEQ_LEN = 200


def seq_to_ids(seq: str) -> np.ndarray:
    return np.array([NUC_TO_IDX.get(c, 0) for c in seq], dtype=np.int64)


def vectorized_onehot(ids: np.ndarray) -> np.ndarray:
    """Faster one-hot via fancy indexing."""
    B, L = ids.shape
    onehot = np.zeros((B, 4, L), dtype=np.float32)
    b_idx = np.arange(B)[:, None]
    l_idx = np.arange(L)[None, :]
    onehot[b_idx, ids, l_idx] = 1.0
    return onehot


def load_method_seqs(rel_path: str, max_seqs: int, seed: int = 0):
    """Load up to max_seqs sequences from a method CSV. Random subsample if larger."""
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(full_path):
        return None
    df = pd.read_csv(full_path)
    seq_col = "seq" if "seq" in df.columns else df.columns[0]
    seqs = df[seq_col].astype(str).tolist()
    seqs = [s.upper().replace('N', 'A') for s in seqs if len(s) == SEQ_LEN]
    if len(seqs) == 0:
        return None
    if len(seqs) > max_seqs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(seqs), size=max_seqs, replace=False)
        seqs = [seqs[i] for i in idx]
    return seqs
