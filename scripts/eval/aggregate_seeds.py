#!/usr/bin/env python3
"""Aggregate property + novelty + GC metrics across seeds for one cell type.

Reads the `eval_metrics.json` written by the measurement scripts — one per
(method, cell, seed) tuple listed in `scripts.eval.manifest.build_work` — and
emits a per-cell mean ± std table.

Output:

  runs/eval_aggregate_{cell}/eval_mean_std_{cell}.csv

Schema (one row per method):

  method, cell_type, n_seeds_with_eval,
  mu_mean_mean,            mu_mean_std,
  mu_top_mean_mean,        mu_top_mean_std,
  mmd_rbf_mean,            mmd_rbf_std,
  nn_hamming_mean,         nn_hamming_std,
  sliced_wasserstein_mean, sliced_wasserstein_std,
  gc_mean_mean,            gc_mean_std,
  gc_std_mean

Missing JSON files (a method with no samples for that cell line, or an eval
that has not finished) appear as NaN, with `n_seeds_with_eval` reflecting how
many seeds did report.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from scripts.eval.manifest import build_work

# Code lives here; `runs/...` paths are resolved against PROJECT_ROOT, which
# defaults to this repository but can point at a separate run tree.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.environ.get("LOCOGEN_PROJECT_ROOT", REPO_ROOT)

# (out_column, json_path) — extracted from eval_metrics.json with dotted lookup.
METRICS = [
    ("mu_mean",            ("independent_oracle", "independent_mu_mean")),
    ("mu_top_mean",        ("independent_oracle", "independent_mu_top_mean")),
    ("mu_top_5pct_mean",   ("independent_oracle", "independent_mu_top_5pct_mean")),
    ("mu_top_10pct_mean",  ("independent_oracle", "independent_mu_top_10pct_mean")),
    ("mu_top_15pct_mean",  ("independent_oracle", "independent_mu_top_15pct_mean")),
    ("mmd_rbf",            ("feature_dist",       "mmd_rbf")),
    ("nn_hamming",         ("novelty",            "novelty_mean")),
    ("sliced_wasserstein", ("feature_dist",       "sliced_wasserstein")),
    ("gc_mean",            ("gc",                 "gc_mean")),
    ("gc_std",             ("gc",                 "gc_std")),
]

# Extended novelty metrics — read from a sibling `novelty_extended.json`
# emitted by `scripts.eval.compute_novelty_extended`. Flat-key lookup (the
# extended JSON is a single-level dict).
EXTENDED_METRICS = [
    "mmd_rbf_vs_all_train",
    "kmer_jsd_k4_vs_all_train",
    "kmer_jsd_k6_vs_all_train",
    "kmer_cosine_k4_vs_all_train",
    # Cell-stratified reference (parallel set; emitted when cell-cache exists).
    "mmd_rbf_vs_cell_train",
    "kmer_jsd_k4_vs_cell_train",
    "kmer_jsd_k6_vs_cell_train",
    "kmer_cosine_k4_vs_cell_train",
    # Canonical MMD on per-sequence k-mer frequency vectors (model-free).
    "mmd_rbf_kmer_k4_vs_all_train",
    "mmd_rbf_kmer_k6_vs_all_train",
    "mmd_rbf_kmer_k4_vs_cell_train",
    "mmd_rbf_kmer_k6_vs_cell_train",
    # Coverage / density / precision / recall (Naeem et al. 2020).
    # Splits mode-collapse from spread methods that look identical under MMD.
    "precision_kmer_k4_vs_all_train",
    "recall_kmer_k4_vs_all_train",
    "density_kmer_k4_vs_all_train",
    "coverage_kmer_k4_vs_all_train",
    "precision_kmer_k4_vs_cell_train",
    "recall_kmer_k4_vs_cell_train",
    "density_kmer_k4_vs_cell_train",
    "coverage_kmer_k4_vs_cell_train",
    # Intra-set diversity (gen-only, no reference needed).
    "intra_set_nn_mean",
    "intra_set_nn_median",
    "intra_set_nn_iqr",
    "kmer6_vocab_frac",
    "kmer6_set_entropy",
    # Rarity (Han et al., ICCV 2023) — on-manifold sparse-region novelty.
    "rarity_median_vs_all_train",
    "rarity_mean_vs_all_train",
    "rarity_p25_vs_all_train",
    "rarity_p75_vs_all_train",
    "rarity_median_vs_cell_train",
    "rarity_mean_vs_cell_train",
    "rarity_p25_vs_cell_train",
    "rarity_p75_vs_cell_train",
    # Authenticity (Alaa et al., ICML 2022) — non-memorisation fraction.
    "authenticity_vs_all_train",
    "authenticity_vs_cell_train",
    # On-manifold fraction (denominator of the rarity statistic).
    "on_manifold_frac_vs_all_train",
    "on_manifold_frac_vs_cell_train",
    # β-Recall — Diversity (Alaa et al., ICML 2022, Eq 6 + Eq 9).
    "IR_beta_vs_all_train",
    "delta_R_beta_vs_all_train",
    "R_beta_at_0p25_vs_all_train",
    "R_beta_at_0p50_vs_all_train",
    "R_beta_at_0p75_vs_all_train",
    "R_beta_at_1p00_vs_all_train",
    "IR_beta_vs_cell_train",
    "delta_R_beta_vs_cell_train",
    "R_beta_at_0p25_vs_cell_train",
    "R_beta_at_0p50_vs_cell_train",
    "R_beta_at_0p75_vs_cell_train",
    "R_beta_at_1p00_vs_cell_train",
]

# DRAKES-paper-style metrics (read from sibling drakes_paper_metrics.json).
DRAKES_PAPER_METRICS = [
    "pred_activity_median",
    "atac_acc_pct",
    "kmer3_corr",
    "jaspar_corr",
    "app_log_lik_median",
]


def _read_one(out_dir: str):
    """Return dict of metric → float, or None if eval JSON missing/malformed.

    Also merges in `novelty_extended.json` from the same directory if present
    (skipped silently otherwise → those columns become NaN downstream).
    """
    path = os.path.join(out_dir, "eval_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    out = {}
    for col, (top, leaf) in METRICS:
        v = d.get(top, {}).get(leaf, None)
        out[col] = float(v) if v is not None else np.nan

    ext_path = os.path.join(out_dir, "novelty_extended.json")
    if os.path.exists(ext_path):
        with open(ext_path) as f:
            ext = json.load(f)
        for col in EXTENDED_METRICS:
            v = ext.get(col, None)
            out[col] = float(v) if v is not None else np.nan
    else:
        for col in EXTENDED_METRICS:
            out[col] = np.nan

    drakes_path = os.path.join(out_dir, "drakes_paper_metrics.json")
    if os.path.exists(drakes_path):
        with open(drakes_path) as f:
            dp = json.load(f)
        for col in DRAKES_PAPER_METRICS:
            v = dp.get(col, None)
            out[col] = float(v) if v is not None else np.nan
    else:
        for col in DRAKES_PAPER_METRICS:
            out[col] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell_type", required=True, choices=["hepg2", "k562", "sknsh"])
    ap.add_argument("--out_dir", default=None,
                    help="Default: runs/eval_aggregate_{cell}/")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        PROJECT_ROOT, f"runs/eval_aggregate_{args.cell_type}")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    n_missing = 0
    for w in build_work():
        if w["cell"] != args.cell_type:
            continue
        m = _read_one(w["out_dir"])
        if m is None:
            n_missing += 1
            print(f"[skip-missing] {w['label']:30s} seed={w['seed']}  -> {w['out_dir']}/eval_metrics.json")
            continue
        rows.append({
            "method": w["label"],
            "cell_type": args.cell_type,
            "seed": w["seed"],
            **m,
        })

    if not rows:
        raise SystemExit(f"No eval JSONs found for cell={args.cell_type}. "
                         f"Did the eval array finish?")

    full = pd.DataFrame(rows)
    print(f"[load] cell={args.cell_type}  n_present={len(full)}  n_missing={n_missing}")

    metric_cols = (
        [c for (c, _) in METRICS]
        + list(EXTENDED_METRICS)
        + list(DRAKES_PAPER_METRICS)
    )
    grp = full.groupby(["method", "cell_type"], sort=False)

    # Aggregate. mean+std for everything; gc_std additionally aggregated as
    # cross-seed mean (per-seed gc_std is itself a property of the set, so
    # cross-seed std would be a std-of-std — keep mean only).
    agg_dict = {}
    for c in metric_cols:
        if c == "gc_std":
            agg_dict[c] = ["mean"]
        else:
            agg_dict[c] = ["mean", "std", "count"]
    out = grp.agg(agg_dict).reset_index()
    out.columns = [c if not isinstance(c, tuple) else
                   (f"{c[0]}_{c[1]}" if c[1] else c[0]) for c in out.columns]

    # Add a cleaner n_seeds_with_eval column (use mu_mean_count as proxy).
    out = out.rename(columns={"mu_mean_count": "n_seeds_with_eval"})
    # Drop redundant per-metric count cols (all equal to n_seeds_with_eval
    # since every metric is read from the same JSON).
    drop_cols = [c for c in out.columns
                 if c.endswith("_count") and c != "n_seeds_with_eval"]
    out = out.drop(columns=drop_cols)

    out_path = os.path.join(out_dir, f"eval_mean_std_{args.cell_type}.csv")
    out.to_csv(out_path, index=False)
    print(f"[write] {out_path}  ({len(out):,} rows)")

    pretty_cols = ["method", "n_seeds_with_eval",
                   "mu_mean_mean", "mu_mean_std",
                   "mmd_rbf_mean", "mmd_rbf_std",
                   "nn_hamming_mean", "nn_hamming_std"]
    print(f"\n=== {args.cell_type} — eval mean ± std across seeds ===")
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", lambda x: f"{x:.4f}"):
        print(out[pretty_cols].to_string(index=False))


if __name__ == "__main__":
    main()
