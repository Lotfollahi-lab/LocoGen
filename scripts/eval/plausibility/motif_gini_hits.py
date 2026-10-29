"""
Per-method JASPAR motif diversity audit.

Uses `scan_jaspar_motifs` from `locogen.utils.jaspar` to count per-motif hits
across each method's generated sequences. Then computes:
  - motif_diversity_shannon : Shannon entropy of per-motif hit distribution
                              (normalized by log(num_motifs); 1 = uniform,
                              0 = one motif dominates)
  - motif_gini              : Gini coefficient of the same distribution
                              (0 = uniform, 1 = total concentration)
  - top10_motifs            : top-10 JASPAR IDs by hit count, comma-separated

Question motivating: SVDD M=30 has motif_hits_per_seq_mean = 629 vs train 426
(+48%). Is this diverse TF enrichment or GC-rich motif over-amplification?
Shannon entropy < train AND Gini > train would indicate concentration
(artifact); Shannon ≈ train would indicate diverse enrichment (genuine).

Outputs (under --out_dir):
  per_motif.csv         long format: method, motif_id, hit_rate (hits per seq)
  diversity_summary.csv one row per method with shannon, gini, top10

`--label` and `--csv` are repeatable and consumed pairwise, so one invocation
scans several methods into a single pair of output CSVs.

    python -m scripts.eval.plausibility.motif_gini_hits \\
        --label "LocoGen (lam=0.7)" --csv runs/<run>/sequences.csv \\
        --label Train --csv runs/train_subsample_hepg2_5k.csv \\
        --out_dir runs/audit_motif_diversity_hepg2_seed0
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

from locogen.metrics.motif_usage import gini, hits_per_seq_stats, shannon_entropy
from locogen.utils.jaspar import scan_jaspar_motifs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", required=True,
                    help="Method label; pair with matching --csv (repeatable, in order).")
    ap.add_argument("--csv", action="append", required=True,
                    help="Sequence CSV path. Repeatable; same order as --label.")
    ap.add_argument("--out_dir", required=True,
                    help="Directory for per_motif.csv and diversity_summary.csv.")
    ap.add_argument("--max_motifs", type=int, default=None,
                    help="Cap motifs scanned (debug only; default = all CORE vertebrates ~841).")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--save_per_seq", action="store_true",
                    help="Persist per-sequence × motif hit matrix as NPZ "
                         "under {out_dir}/per_seq/ for within-sequence "
                         "concentration analysis.")
    args = ap.parse_args()

    if len(args.label) != len(args.csv):
        raise SystemExit(f"--label/--csv count mismatch: "
                         f"{len(args.label)} vs {len(args.csv)}")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[motif] {len(args.csv)} method(s) -> {args.out_dir}")

    long_rows = []
    summary_rows = []

    for label, path in zip(args.label, args.csv):
        if not os.path.exists(path):
            print(f"[skip] {label}: missing {path}")
            continue
        df = pd.read_csv(path)
        if "seq" not in df.columns:
            raise RuntimeError(f"{path} missing 'seq' column (cols: {list(df.columns)})")
        seqs = df["seq"].astype(str).tolist()
        n = len(seqs)
        print(f"[scan] {label}: n={n} from {path}", flush=True)

        if args.save_per_seq:
            hits_per_seq, motif_freq, motif_ids, per_seq_hits = scan_jaspar_motifs(
                seqs, max_motifs=args.max_motifs, return_per_seq=True)
            slug = re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_').lower()
            ps_dir = os.path.join(args.out_dir, "per_seq")
            os.makedirs(ps_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(ps_dir, f"{slug}.npz"),
                hits=per_seq_hits,
                motif_ids=np.asarray(motif_ids, dtype=object),
                n_seqs=np.int32(n),
                method=np.asarray(label, dtype=object),
            )
            print(f"  [per_seq] saved {ps_dir}/{slug}.npz "
                  f"shape={per_seq_hits.shape} dtype={per_seq_hits.dtype}",
                  flush=True)
        else:
            hits_per_seq, motif_freq, motif_ids = scan_jaspar_motifs(
                seqs, max_motifs=args.max_motifs)
        # `motif_freq[j]` = total hits for motif j averaged over n seqs

        # Long format per-motif rows
        for mid, hr in zip(motif_ids, motif_freq):
            long_rows.append({"method": label, "motif_id": mid, "hits_per_seq": float(hr)})

        # Summary
        sh = shannon_entropy(motif_freq)
        gi = gini(motif_freq)
        order = np.argsort(motif_freq)[::-1][:args.top_k]
        top_ids = [motif_ids[i] for i in order]
        top_rates = [float(motif_freq[i]) for i in order]
        summary_rows.append({
            "method": label,
            "n_seqs": n,
            "motif_hits_per_seq_mean":
                hits_per_seq_stats(hits_per_seq)["motif_hits_per_seq_mean"],
            "motif_diversity_shannon": sh,
            "motif_gini": gi,
            "top10_motifs": ",".join(top_ids),
            "top10_rates":  ",".join(f"{r:.2f}" for r in top_rates),
            "num_motifs_scanned": len(motif_ids),
        })
        print(f"  hits/seq={hits_per_seq.mean():.1f}  shannon={sh:.4f}  gini={gi:.4f}  top1={top_ids[0]}({top_rates[0]:.1f})",
              flush=True)

    long_df = pd.DataFrame(long_rows)
    sum_df = pd.DataFrame(summary_rows)
    long_path = os.path.join(args.out_dir, "per_motif.csv")
    sum_path  = os.path.join(args.out_dir, "diversity_summary.csv")
    long_df.to_csv(long_path, index=False)
    sum_df.to_csv(sum_path, index=False)
    print(f"\n[saved] {long_path}  ({len(long_df)} rows)")
    print(f"[saved] {sum_path}    ({len(sum_df)} rows)")
    print()
    print(sum_df[["method", "motif_hits_per_seq_mean", "motif_diversity_shannon", "motif_gini"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
