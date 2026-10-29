"""Novelty axis: Authenticity and Rarity.

Authenticity (Alaa et al., 2022) is the fraction of generated sequences that
lie farther from their nearest training neighbour than that neighbour's own
leave-one-out 1-NN distance, so a high value means little memorisation.
Rarity (Han et al., 2023) is the radius of the smallest training k-NN ball
containing a generated sequence, reported only for on-manifold samples, so a
high value means the sample sits in a sparse region of training density.

Both are computed on per-sequence 4-mer frequency vectors against a cached
reference set, built once with --build_cache_only / --build_cell_cache.

    python -m scripts.eval.novelty.authenticity_rarity --build_cache_only
    python -m scripts.eval.novelty.authenticity_rarity \\
        --generated runs/<run>/sequences.csv \\
        --cell hepg2 --out_dir runs/<run>/eval
"""
import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from locogen.utils.seq_io import load_train_sequences
from locogen.metrics.authenticity import compute_authenticity
from locogen.metrics.kmer_features import kth_nn_distance, per_seq_kmer_freq
from locogen.metrics.rarity import compute_rarity_score
from locogen.utils.seq_stats import group_kmer_distribution


# Code lives here; `runs/...` paths are resolved against PROJECT_ROOT, which
# defaults to this repository but can point at a separate run tree.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
PROJECT_ROOT = os.environ.get("LOCOGEN_PROJECT_ROOT", REPO_ROOT)
# Gosai reference set. Same default as the Stage 1 configs; see the README for
# the symlink that points it at the copy inside the DRAKES data archive.
DEFAULT_REF = os.path.join(PROJECT_ROOT, "data/gosai_all.csv")
CACHE_DIR = os.path.join(PROJECT_ROOT, "runs/cache/novelty_ref")
DEFAULT_N_REF = 10_000  # subsample size from gosai_all
KMER_SIZES = (4, 6)
EPS = 1e-12

# Cell-specific reference: top-activity training subsample per cell.
# These are 5k-row CSVs with a single `seq` column; built separately and
# stored alongside the project, used as the cell-stratified novelty reference.
CELL_REF = {
    "hepg2": os.path.join(PROJECT_ROOT, "runs/train_subsample_hepg2_5k.csv"),
    "k562":  os.path.join(PROJECT_ROOT, "runs/train_subsample_k562_5k.csv"),
    "sknsh": os.path.join(PROJECT_ROOT, "runs/train_subsample_sknsh_5k.csv"),
}


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------
def _cache_paths(n_ref: int) -> Dict[str, str]:
    base = os.path.join(CACHE_DIR, f"all_train_{n_ref}")
    return {
        "kmer_k4":  f"{base}_kmer_k4.npy",
        "kmer_k6":  f"{base}_kmer_k6.npy",
        "per_seq_kmer_k4": f"{base}_per_seq_kmer_k4.npz",
        "per_seq_kmer_k6": f"{base}_per_seq_kmer_k6.npz",
        "kmer_k4_real_kth": f"{base}_kmer_k4_real_kth.npy",
        "train_nn1_dist_k4": f"{base}_train_nn1_dist_k4.npy",
        "seqs":     f"{base}_seqs.txt",
    }


def _cell_cache_paths(cell: str) -> Dict[str, str]:
    base = os.path.join(CACHE_DIR, f"cell_train_{cell}")
    return {
        "kmer_k4":  f"{base}_kmer_k4.npy",
        "kmer_k6":  f"{base}_kmer_k6.npy",
        "per_seq_kmer_k4": f"{base}_per_seq_kmer_k4.npz",
        "per_seq_kmer_k6": f"{base}_per_seq_kmer_k6.npz",
        "kmer_k4_real_kth": f"{base}_kmer_k4_real_kth.npy",
        "train_nn1_dist_k4": f"{base}_train_nn1_dist_k4.npy",
        "seqs":     f"{base}_seqs.txt",
    }


def _load_seqs_from_csv(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path, usecols=["seq"])
    return [s.upper() for s in df["seq"].tolist()]


def build_cell_cache() -> None:
    """Build per-cell reference cache (HepG2 / K562 / SK-N-SH) from the
    top-activity training subsample CSVs. Idempotent."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    for cell, ref_csv in CELL_REF.items():
        paths = _cell_cache_paths(cell)
        needs_kmer = any(not os.path.exists(paths[f"kmer_k{k}"])
                         for k in KMER_SIZES)
        needs_per_seq = any(not os.path.exists(paths[f"per_seq_kmer_k{k}"])
                            for k in KMER_SIZES)
        needs_real_kth = not os.path.exists(paths["kmer_k4_real_kth"])
        needs_nn1 = not os.path.exists(paths["train_nn1_dist_k4"])
        if not (needs_kmer or needs_per_seq or needs_real_kth or needs_nn1):
            print(f"[cache] cell={cell}: all artefacts present")
            continue
        seqs = _load_seqs_from_csv(ref_csv)
        print(f"[cache] cell={cell}: {len(seqs)} ref seqs from {ref_csv}")
        with open(paths["seqs"], "w") as f:
            for s in seqs:
                f.write(s + "\n")
        if needs_kmer:
            for k in KMER_SIZES:
                kpath = paths[f"kmer_k{k}"]
                if os.path.exists(kpath):
                    continue
                print(f"[cache] cell={cell} k={k} (set-aggregated)")
                counts = group_kmer_distribution(seqs, k)
                np.save(kpath, counts)
        if needs_per_seq:
            for k in KMER_SIZES:
                ppath = paths[f"per_seq_kmer_k{k}"]
                if os.path.exists(ppath):
                    continue
                print(f"[cache] cell={cell} per-seq k={k}")
                ps = per_seq_kmer_freq(seqs, k)
                np.savez_compressed(ppath, per_seq=ps)
        # k-NN radii in cell-stratified per-seq k-mer (k=4) feature space.
        if not os.path.exists(paths["kmer_k4_real_kth"]):
            ps_k4 = np.load(paths["per_seq_kmer_k4"])["per_seq"]
            print(f"[cache] cell={cell} computing kmer_k4 real_kth")
            radii = kth_nn_distance(ps_k4, k=5)
            np.save(paths["kmer_k4_real_kth"], radii)
        # Leave-one-out 1-NN distance per train seq, for Alaa 2022 Authenticity.
        if not os.path.exists(paths["train_nn1_dist_k4"]):
            ps_k4 = np.load(paths["per_seq_kmer_k4"])["per_seq"]
            print(f"[cache] cell={cell} computing kmer_k4 train_nn1_dist")
            nn1 = kth_nn_distance(ps_k4, k=1)
            np.save(paths["train_nn1_dist_k4"], nn1)

    print("[cache] cell-stratified caches done")


def load_cell_cache(cell: str) -> Dict:
    paths = _cell_cache_paths(cell)
    out = {
        "kmer_k4": np.load(paths["kmer_k4"]),
        "kmer_k6": np.load(paths["kmer_k6"]),
    }
    for k in KMER_SIZES:
        ppath = paths[f"per_seq_kmer_k{k}"]
        if os.path.exists(ppath):
            out[f"per_seq_kmer_k{k}"] = np.load(ppath)["per_seq"]
    if os.path.exists(paths["kmer_k4_real_kth"]):
        out["kmer_k4_real_kth"] = np.load(paths["kmer_k4_real_kth"])
    if os.path.exists(paths["train_nn1_dist_k4"]):
        out["train_nn1_dist_k4"] = np.load(paths["train_nn1_dist_k4"])
    return out


def _subsample_ref_seqs(ref_csv: str, n_ref: int, seed: int = 0) -> list[str]:
    """Random subsample n_ref sequences from gosai_all.csv (deterministic)."""
    df = pd.read_csv(ref_csv, usecols=["seq"])
    n_total = len(df)
    if n_ref >= n_total:
        return [s.upper() for s in df["seq"].tolist()]
    idx = np.random.default_rng(seed).choice(n_total, size=n_ref, replace=False)
    return [df.iloc[i]["seq"].upper() for i in sorted(idx.tolist())]


# ---------------------------------------------------------------------------
# Reference cache builder
# ---------------------------------------------------------------------------
def build_cache(ref_csv: str, n_ref: int) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = _cache_paths(n_ref)

    needs_kmer = any(not os.path.exists(paths[f"kmer_k{k}"]) for k in KMER_SIZES)
    needs_per_seq = any(not os.path.exists(paths[f"per_seq_kmer_k{k}"])
                        for k in KMER_SIZES)
    needs_real_kth = not os.path.exists(paths["kmer_k4_real_kth"])
    needs_nn1 = not os.path.exists(paths["train_nn1_dist_k4"])

    if not (needs_kmer or needs_per_seq or needs_real_kth or needs_nn1):
        print(f"[cache] all reference artefacts present in {CACHE_DIR}")
        return

    print(f"[cache] subsampling {n_ref} seqs from {ref_csv}")
    seqs = _subsample_ref_seqs(ref_csv, n_ref, seed=0)
    print(f"[cache] {len(seqs)} reference seqs")

    with open(paths["seqs"], "w") as f:
        for s in seqs:
            f.write(s + "\n")

    if needs_kmer:
        for k in KMER_SIZES:
            kpath = paths[f"kmer_k{k}"]
            if os.path.exists(kpath):
                continue
            print(f"[cache] computing reference k-mer k={k} (4^{k}={4**k} dims)")
            counts = group_kmer_distribution(seqs, k)
            np.save(kpath, counts)
            print(f"[cache] wrote {kpath}")

    if needs_per_seq:
        for k in KMER_SIZES:
            ppath = paths[f"per_seq_kmer_k{k}"]
            if os.path.exists(ppath):
                continue
            print(f"[cache] computing per-seq k-mer matrix k={k} "
                  f"(N={len(seqs)} × {4**k} dims)")
            ps = per_seq_kmer_freq(seqs, k)
            np.savez_compressed(ppath, per_seq=ps)
            print(f"[cache] wrote {ppath}  shape={ps.shape}")

    # Pre-compute k-NN radii in reference k-mer (k=4) feature space for
    # coverage / density. Computed once here so per-task PRDC is fast.
    if not os.path.exists(paths["kmer_k4_real_kth"]):
        ps_k4 = np.load(paths["per_seq_kmer_k4"])["per_seq"]
        print(f"[cache] computing kmer_k4 real_kth (k=5) over {ps_k4.shape[0]} ref seqs")
        radii = kth_nn_distance(ps_k4, k=5)
        np.save(paths["kmer_k4_real_kth"], radii)
        print(f"[cache] wrote {paths['kmer_k4_real_kth']}  shape={radii.shape}")

    # Leave-one-out 1-NN distance per train seq, for Alaa 2022 Authenticity.
    if not os.path.exists(paths["train_nn1_dist_k4"]):
        ps_k4 = np.load(paths["per_seq_kmer_k4"])["per_seq"]
        print(f"[cache] computing kmer_k4 train_nn1_dist over {ps_k4.shape[0]} ref seqs")
        nn1 = kth_nn_distance(ps_k4, k=1)
        np.save(paths["train_nn1_dist_k4"], nn1)
        print(f"[cache] wrote {paths['train_nn1_dist_k4']}  shape={nn1.shape}")

    print("[cache] done")


def load_cache(n_ref: int) -> Dict:
    paths = _cache_paths(n_ref)
    out = {}
    for k in KMER_SIZES:
        out[f"kmer_k{k}"] = np.load(paths[f"kmer_k{k}"])
        ppath = paths[f"per_seq_kmer_k{k}"]
        if os.path.exists(ppath):
            out[f"per_seq_kmer_k{k}"] = np.load(ppath)["per_seq"]
    if os.path.exists(paths["kmer_k4_real_kth"]):
        out["kmer_k4_real_kth"] = np.load(paths["kmer_k4_real_kth"])
    if os.path.exists(paths["train_nn1_dist_k4"]):
        out["train_nn1_dist_k4"] = np.load(paths["train_nn1_dist_k4"])
    return out


# ---------------------------------------------------------------------------
# Per-task compute
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Coverage / Density / Precision / Recall  (Naeem et al. 2020, "Reliable
# Fidelity and Diversity Metrics for Generative Models")
#
# Operates on per-sequence k-mer-frequency vectors. Distinguishes mode-collapse
# from true novelty: a method that produces 640 near-duplicates near a single
# train mode has the same MMD-vs-train as a method that spreads across the
# manifold; coverage / recall split them.
#
# Conventions (matching the reference `prdc` package):
#   real = train reference, fake = generated set
#   real_kth[i]  = distance from real_i to its k-th NN in real (excluding self)
#   fake_kth[i]  = distance from fake_i to its k-th NN in fake (excluding self)
#   d[i, j]      = euclidean distance between real_i and fake_j
#
#   precision = mean over fakes of (any real has fake within its real_kth ball)
#   recall    = mean over reals of (any fake has real within its fake_kth ball)
#   density   = mean over fakes of (#real balls covering fake) / k
#   coverage  = mean over reals of (any fake within its real_kth ball)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Rarity Score  (Han et al., ICCV 2023, "Rarity Score: Investigating the
# Rarity of the Generated Samples").
#
# For each on-manifold gen sample x_g, the rarity is the smallest train
# k-NN-ball radius among the train balls that contain x_g. Off-manifold
# samples (no containing ball) are excluded from the radius statistics.
# `on_manifold` here matches the Naeem precision predicate by construction
# and uses k=5 for consistency with `compute_prdc`.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# β-Recall  (Alaa et al., ICML 2022, Eq 9) — Diversity component of the
# (α-Precision, β-Recall, Authenticity) triple.
#
# For each real X̃_r,i, R̂_β,i = 1 iff there exists a synthetic X̃_g,j
# satisfying BOTH:
#   (i)  ||X̃_g,j − c_g|| ≤ r̂_β       (synthetic in β-typical region of P_g)
#   (ii) ||X̃_g,j − X̃_r,i|| ≤ NND_k(X̃_r,i)   (synthetic in real i's k-NN ball)
# where c_g = mean of synthetic embeddings, r̂_β = β-quantile of synthetics'
# distances to c_g, and NND_k uses the same k=5 as Naeem PRDC.
#
# R_β curve is integrated to a single scalar (Eq 6):
#   ΔR_β = ∫₀¹ |R_β − β| dβ ;  IR_β = 1 − 2·ΔR_β ∈ [0,1].
# IR_β = 1 ⟺ P_g = P_r (Theorem 1). Reduces to Naeem Coverage at β = 1.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Authenticity  (Alaa et al., ICML 2022, "How Faithful is your Synthetic
# Data? Sample-level Metrics for Evaluating and Auditing Generative Models").
#
# A gen sample is "authentic" if its distance to its nearest train neighbour
# exceeds that train neighbour's own (leave-one-out) 1-NN distance to other
# train points. Mean over gen ∈ [0,1]: low ⇒ memorisation, high ⇒ non-copying.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Intra-set diversity  (within the generated set itself, no reference)
#
# Complements coverage/density: tells us whether the gen set itself is
# concentrated or spread, independent of the training distribution. Useful
# diagnostic for mode-collapse even if no reference is available.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Per-task compute
# ---------------------------------------------------------------------------
def compute_one(gen_csv: str, cache: Dict, cell_cache: Dict = None) -> Dict:
    """Authenticity and Rarity for one generated-sequence CSV.

    Both operate on per-sequence 4-mer frequency vectors, compared against a
    cached reference set: the uniform training subsample (`*_vs_all_train`)
    and, when a cell-stratified cache exists, the per-cell top-activity
    subsample (`*_vs_cell_train`).
    """
    if not os.path.exists(gen_csv):
        return {"_skip": f"missing {gen_csv}"}

    gen_seqs = load_train_sequences(gen_csv)  # same loader, just reads `seq` column
    if not gen_seqs:
        return {"_skip": f"no rows in {gen_csv}"}

    gen_ps_k4 = per_seq_kmer_freq(gen_seqs, 4)
    out = {
        "n_gen": int(len(gen_seqs)),
        "ref_path": DEFAULT_REF,
    }

    ref_ps_k4 = cache.get("per_seq_kmer_k4")
    if ref_ps_k4 is not None:
        out["n_ref"] = int(ref_ps_k4.shape[0])
        real_kth = cache.get("kmer_k4_real_kth")
        if real_kth is not None:
            for key, val in compute_rarity_score(ref_ps_k4, gen_ps_k4, real_kth).items():
                out[f"{key}_vs_all_train"] = val
        train_nn1 = cache.get("train_nn1_dist_k4")
        if train_nn1 is not None:
            for key, val in compute_authenticity(ref_ps_k4, gen_ps_k4, train_nn1).items():
                out[f"{key}_vs_all_train"] = val

    if cell_cache is not None:
        ref_ps_k4_c = cell_cache.get("per_seq_kmer_k4")
        if ref_ps_k4_c is not None:
            out["n_ref_cell"] = int(ref_ps_k4_c.shape[0])
            real_kth_c = cell_cache.get("kmer_k4_real_kth")
            if real_kth_c is not None:
                for key, val in compute_rarity_score(
                        ref_ps_k4_c, gen_ps_k4, real_kth_c).items():
                    out[f"{key}_vs_cell_train"] = val
            train_nn1_c = cell_cache.get("train_nn1_dist_k4")
            if train_nn1_c is not None:
                for key, val in compute_authenticity(
                        ref_ps_k4_c, gen_ps_k4, train_nn1_c).items():
                    out[f"{key}_vs_cell_train"] = val
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
REQUIRED_KEYS = {
    "rarity_median_vs_all_train",
    "rarity_median_vs_cell_train",
    "authenticity_vs_all_train",
    "authenticity_vs_cell_train",
    "on_manifold_frac_vs_all_train",
    "on_manifold_frac_vs_cell_train",
}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build_cache_only", action="store_true",
                   help="One-time: build uniform-train reference cache, then exit.")
    g.add_argument("--build_cell_cache", action="store_true",
                   help="One-time: build cell-stratified reference caches "
                        "(HepG2/K562/SK-N-SH), then exit.")
    g.add_argument("--generated", type=str,
                   help="Generated-sequence CSV to score (one run's sequences.csv).")

    ap.add_argument("--cell", choices=("hepg2", "k562", "sknsh"),
                    help="Cell line whose stratified reference cache to compare "
                         "against, for the *_vs_cell_train metrics. Required with "
                         "--generated.")
    ap.add_argument("--out_dir", type=str,
                    help="Directory to write novelty_extended.json into. "
                         "Required with --generated.")
    ap.add_argument("--ref_csv", default=DEFAULT_REF)
    ap.add_argument("--n_ref", type=int, default=DEFAULT_N_REF)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing novelty_extended.json.")
    args = ap.parse_args()

    if args.build_cache_only:
        build_cache(args.ref_csv, args.n_ref)
        return
    if args.build_cell_cache:
        build_cell_cache()
        return

    missing = [f for f in ("cell", "out_dir") if getattr(args, f) is None]
    if missing:
        raise SystemExit("--generated requires "
                         + " and ".join(f"--{f}" for f in missing))

    out_path = os.path.join(args.out_dir, "novelty_extended.json")
    if os.path.exists(out_path) and not args.force:
        try:
            with open(out_path) as f:
                existing = json.load(f)
            if REQUIRED_KEYS.issubset(existing.keys()):
                print(f"[skip] exists with all metrics: {out_path}")
                return
            print(f"[refresh] missing keys "
                  f"{sorted(REQUIRED_KEYS - existing.keys())} — re-computing")
        except Exception:
            pass  # malformed JSON, just re-compute

    cache = load_cache(args.n_ref)
    cell_cache = None
    try:
        cell_cache = load_cell_cache(args.cell)
    except FileNotFoundError:
        print(f"[warn] no cell cache for {args.cell}; "
              "reporting *_vs_all_train only")

    res = compute_one(args.generated, cache, cell_cache=cell_cache)
    if "_skip" in res:
        print(f"[skip] {res['_skip']}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[novelty] saved -> {out_path}")
    for k in sorted(res):
        if k.startswith(("authenticity", "rarity_median", "on_manifold")):
            print(f"  {k} = {res[k]}")


if __name__ == "__main__":
    main()
