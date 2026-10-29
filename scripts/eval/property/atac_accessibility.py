"""Property axis, metric 2 of 2: ATAC_acc (cell-aware).

Scores generated sequences with the DRAKES-distributed
`binary_atac_cell_lines.ckpt` — a 7-head classifier predicting whether DNA
looks like it came from each of 7 cell lines' open chromatin regions
(DRAKES Section 6.2 "ATAC-acc").

The head for the requested cell line is read from `CELL_HEAD` below, which
follows the checkpoint's own metadata
(`hyper_parameters.data_params.tasks.name`) and was verified empirically by
per-head Pearson correlation against the gosai labels:

    HepG2  -> head 1   (corr +0.292 vs hepg2 label, max)
    K562   -> head 4   (corr +0.323 vs k562 label, max)
    SK-N-SH -> head 5  (corr +0.099 vs sknsh label, max)

The head must not be inferred from the argmax distribution over sequences:
head 4 (K562) fires most often even on HepG2-derived enhancers, so an
argmax-inference routine picks it for every cell line. That is a
population-mode artefact, not the target head.

    python -m scripts.eval.property.atac_accessibility \\
        --cell hepg2 \\
        --label "LocoGen HepG2 seed0" --csv runs/<run>/sequences.csv \\
        --out_csv runs/atac_cell_aware/hepg2_locogen.csv

`--label` and `--csv` are repeatable and consumed pairwise, so one
invocation can score several methods into a single output CSV.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from locogen.utils.seq_io import load_method_seqs, seq_to_ids, vectorized_onehot

# DRAKES 7-head binary ATAC classifier, shipped in the DRAKES data archive.
ATAC_CKPT = os.path.join(
    os.environ.get("LOCOGEN_DRAKES_DATA", "./data/DRAKES_data/data_and_model/"),
    "mdlm/gosai_data/binary_atac_cell_lines.ckpt",
)

CELL_HEAD = {
    "hepg2": 1,
    "k562":  4,
    "sknsh": 5,
}
TASK_NAMES = ["GM12878", "HepG2", "IMR90", "Jurkat", "K562", "SKNSH", "WTC11"]


def load_atac_model(device):
    """Load DRAKES 7-head binary ATAC classifier."""
    from locogen.utils.grelu_compat import patch_grelu_lightning_compat
    patch_grelu_lightning_compat()
    from grelu.lightning import LightningModel

    print(f"[atac] loading {ATAC_CKPT}")
    model = LightningModel.load_from_checkpoint(ATAC_CKPT, map_location=device)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def score_atac(seq_ids: np.ndarray, atac_model, device, batch_size: int = 512):
    """(N, L) int → (N, 7) ATAC head outputs."""
    onehot = vectorized_onehot(seq_ids)
    out_chunks = []
    for s in range(0, onehot.shape[0], batch_size):
        x = torch.from_numpy(onehot[s:s + batch_size]).to(device)
        out = atac_model(x)
        if out.dim() == 3:
            out = out.mean(dim=-1)
        out_chunks.append(out.float().cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", choices=sorted(CELL_HEAD.keys()), required=True,
                   help="Target cell line (selects ATAC head).")
    p.add_argument("--label", action="append", required=True,
                   help="Method label; pair with matching --csv (repeatable, in order).")
    p.add_argument("--csv", action="append", required=True,
                   help="Sequence CSV path. Repeatable; same order as --label.")
    p.add_argument("--n_seqs", type=int, default=64,
                   help="Sequences per CSV (matches HepG2 audit's σ_within).")
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()

    if len(args.label) != len(args.csv):
        raise SystemExit(f"--label/--csv count mismatch: "
                         f"{len(args.label)} vs {len(args.csv)}")

    head = CELL_HEAD[args.cell]
    head_name = TASK_NAMES[head]
    print(f"[cell-aware] cell={args.cell}  head={head}  task_name={head_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    atac_model = load_atac_model(device)

    mean_col = f"atac_{args.cell}_mean"
    std_col  = f"atac_{args.cell}_std"
    acc_col  = f"argmax_acc_{args.cell}"

    rows = []
    for label, csv_path in zip(args.label, args.csv):
        seqs = load_method_seqs(csv_path, max_seqs=args.n_seqs)
        if seqs is None:
            print(f"  [skip] {label}: missing or empty {csv_path}")
            continue
        sid = np.array([seq_to_ids(s) for s in seqs], dtype=np.int64)
        atac = score_atac(sid, atac_model, device)
        head_scores = atac[:, head]
        row = {
            "label":   label,
            "csv":     csv_path,
            "cell":    args.cell,
            "head":    head,
            "n_seqs":  len(seqs),
            mean_col:  float(head_scores.mean()),
            std_col:   float(head_scores.std()),
            acc_col:   float((atac.argmax(axis=1) == head).mean()),
        }
        rows.append(row)
        print(f"  [{label}] {mean_col} = {row[mean_col]:.3f} "
              f"± {row[std_col]:.3f}  (argmax-acc={row[acc_col]:.3f}, "
              f"n={row['n_seqs']})")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\n[write] {args.out_csv}")


if __name__ == "__main__":
    main()
