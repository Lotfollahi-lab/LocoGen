"""Property axis, metric 1 of 2: Pred-activity.

Scores generated sequences with the DRAKES-distributed `reward_oracle_eval`
checkpoint, a frozen 3-head grelu Enformer regressor over
[hepg2, k562, sknsh]. This oracle is never used for guidance or fine-tuning
by any method, which is what makes the reported activity non-circular; the
guidance oracle is `reward_oracle_ft` and must never be passed here.

GC content is reported alongside, as a descriptive statistic backing the
composition-drift diagnostic figure.

    python -m scripts.eval.property.pred_activity \\
        --generated runs/<run>/sequences.csv \\
        --train_data runs/train_subsample_hepg2_5k.csv \\
        --oracle_target_idx 0 --out_dir runs/<run>/eval
"""
import argparse
import json
import os

import numpy as np
import torch

from locogen.utils.seq_io import (
    gc_content,
    load_sequences_csv,
    load_train_sequences,
    seqs_to_tensor,
)

# Cap on training sequences pushed through the oracle for
# `mu_wasserstein_vs_train`, and the RNG seed used to subsample them. Both
# match the research-tree defaults, so the metric is unchanged.
DEFAULT_ORACLE_MAX_TRAIN = 5000
TRAIN_SUBSET_SEED = 0


def load_independent_oracle(ckpt_path: str, device):
    """Load the DRAKES-distributed grelu LightningModel used as external oracle.

    The checkpoint was saved via grelu's LightningModel wrapper around
    EnformerPretrainedModel (3-task regression: hepg2, k562, sknsh).
    """
    # DRAKES' ckpt nests data_params/model_params/train_params inside
    # hyper_parameters, but newer grelu expects them at the top level and
    # also reads `performance` directly. Patch before the import.
    from locogen.utils.grelu_compat import patch_grelu_lightning_compat
    patch_grelu_lightning_compat()
    from grelu.lightning import LightningModel
    model = LightningModel.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _seqs_to_onehot(tensor: torch.Tensor) -> torch.Tensor:
    """(B, L) int -> (B, 4, L) float one-hot, the layout grelu models expect."""
    return torch.nn.functional.one_hot(tensor, num_classes=4).permute(0, 2, 1).float()


def score_with_oracle(tensor: torch.Tensor, oracle, device, target_idx: int,
                      batch_size: int = 128) -> np.ndarray:
    """Run the independent oracle over (B, L) token IDs and return (N,) predictions
    for the requested target_idx column (0=hepg2, 1=k562, 2=sknsh)."""
    preds = []
    with torch.no_grad():
        for i in range(0, tensor.size(0), batch_size):
            chunk = _seqs_to_onehot(tensor[i:i + batch_size].to(device))
            out = oracle(chunk)
            if out.dim() == 3:
                out = out.mean(dim=-1)
            preds.append(out[:, target_idx].detach().cpu())
    return torch.cat(preds, 0).numpy()


def compute_independent_oracle_metrics(mu_indep: np.ndarray, top_n: int,
                                       mu_ours: np.ndarray = None,
                                       mu_indep_train: np.ndarray = None,
                                       top_pcts=(5, 10, 15)):
    top_n = min(top_n, len(mu_indep))
    sorted_idx = np.argsort(mu_indep)
    top_idx = sorted_idx[-top_n:]
    metrics = {
        "independent_mu_mean": float(mu_indep.mean()),
        "independent_mu_median": float(np.median(mu_indep)),
        "independent_mu_top_mean": float(mu_indep[top_idx].mean()),
    }
    N = len(mu_indep)
    for p in top_pcts:
        k = max(1, int(np.ceil(N * p / 100.0)))
        idx_p = sorted_idx[-k:]
        metrics[f"independent_mu_top_{p}pct_mean"] = float(mu_indep[idx_p].mean())
        metrics[f"independent_mu_top_{p}pct_n"] = int(k)
    if mu_ours is not None and len(mu_ours) == len(mu_indep):
        corr = float(np.corrcoef(mu_indep, mu_ours)[0, 1])
        metrics["pearson_vs_ours"] = corr
    if mu_indep_train is not None and len(mu_indep_train) > 0:
        metrics["mu_wasserstein_vs_train"] = wasserstein_1d(
            mu_indep, mu_indep_train)
        metrics["independent_mu_train_mean"] = float(mu_indep_train.mean())
    return metrics


def wasserstein_1d(x, y):
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(np.asarray(x), np.asarray(y)))


def compute_gc_stats(gen_seqs):
    gen_gc = [gc_content(s) for s in gen_seqs]
    return {
        "gc_mean": float(np.mean(gen_gc)),
        "gc_std": float(np.std(gen_gc)),
    }


def _resolve_oracle_ckpt(flag_value, allow_circular_only):
    """Resolve which independent-oracle checkpoint to use.

    Priority:
      1. Explicit --oracle_ckpt flag
      2. $DRAKES_EVAL_ORACLE environment variable
      3. None + --allow_circular_only (opt-in diagnostic-only eval)

    Raises SystemExit when none of the above holds, so accidental
    circular-only evaluation is not silently produced.
    """
    if flag_value:
        return flag_value
    env = os.environ.get("DRAKES_EVAL_ORACLE")
    if env:
        return env
    if allow_circular_only:
        return None
    raise SystemExit(
        "[eval] independent oracle required. Pass --oracle_ckpt, set "
        "$DRAKES_EVAL_ORACLE, or pass --allow_circular_only to proceed "
        "with PropertyScorer-only (circular) metrics."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=str, required=True,
                        help="CSV with generated sequences (must have 'seq' column)")
    parser.add_argument("--train_data", type=str, default=None,
                        help="CSV with training sequences; enables "
                             "mu_wasserstein_vs_train")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--top_n", type=int, default=100, help="top-N for top-k metrics")
    parser.add_argument("--seq_col", type=str, default="seq")
    parser.add_argument("--max_train", type=int, default=10000,
                        help="Cap on training seqs read from --train_data")
    parser.add_argument("--cell_type", type=str, default=None,
                        help="If set, keep only generated rows whose 'cell_type' "
                             "column contains this string (case-insensitive). "
                             "Useful for filtering DNA-Diffusion's per-cell-type "
                             "output down to e.g. HepG2 only.")
    parser.add_argument("--oracle_ckpt", type=str, default=None,
                        help="Path to DRAKES-distributed reward_oracle_eval.ckpt "
                             "(grelu Enformer, 3-task). If omitted, the env var "
                             "$DRAKES_EVAL_ORACLE is used.")
    parser.add_argument("--oracle_target_idx", type=int, default=0,
                        help="Which of the oracle's 3 heads to report "
                             "(0=hepg2, 1=k562, 2=sknsh). Default 0.")
    parser.add_argument("--oracle_batch_size", type=int, default=128)
    parser.add_argument("--oracle_max_train", type=int, default=None,
                        help="Cap on training seqs passed through the oracle "
                             f"(default {DEFAULT_ORACLE_MAX_TRAIN}).")
    args = parser.parse_args()

    args.oracle_ckpt = _resolve_oracle_ckpt(args.oracle_ckpt, False)
    if not args.oracle_ckpt or not os.path.exists(args.oracle_ckpt):
        raise SystemExit(
            f"[eval] independent oracle ckpt not found: {args.oracle_ckpt!r}. "
            "Set --oracle_ckpt or $DRAKES_EVAL_ORACLE.")

    print(f"[eval] loading generated sequences from {args.generated}")
    gen_seqs = load_sequences_csv(args.generated, args.seq_col,
                                  cell_type=args.cell_type)
    if args.cell_type:
        print(f"[eval] filtered to cell_type~={args.cell_type!r}")
    print(f"[eval] {len(gen_seqs)} generated sequences loaded")

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_tensor = seqs_to_tensor(gen_seqs)

    train_seqs = None
    if args.train_data and os.path.exists(args.train_data):
        train_seqs = load_train_sequences(args.train_data, args.seq_col,
                                          max_seqs=args.max_train)
        print(f"[eval] {len(train_seqs)} training sequences loaded")

    print(f"[eval] loading independent oracle from {args.oracle_ckpt}")
    oracle = load_independent_oracle(args.oracle_ckpt, device)
    mu_indep = score_with_oracle(
        gen_tensor, oracle, device,
        target_idx=args.oracle_target_idx,
        batch_size=args.oracle_batch_size)

    mu_indep_train = None
    if train_seqs is not None:
        cap = (args.oracle_max_train if args.oracle_max_train is not None
               else DEFAULT_ORACLE_MAX_TRAIN)
        if cap is not None and cap < len(train_seqs):
            idx = np.random.default_rng(TRAIN_SUBSET_SEED).choice(
                len(train_seqs), size=cap, replace=False)
            train_sub = [train_seqs[int(i)] for i in idx]
        else:
            train_sub = train_seqs
        print(f"[eval] scoring {len(train_sub)} training seqs through the oracle")
        mu_indep_train = score_with_oracle(
            seqs_to_tensor(train_sub), oracle, device,
            target_idx=args.oracle_target_idx,
            batch_size=args.oracle_batch_size)

    del oracle
    if device.type == "cuda":
        torch.cuda.empty_cache()

    indep = compute_independent_oracle_metrics(
        mu_indep, args.top_n, mu_ours=None, mu_indep_train=mu_indep_train)
    results["independent_oracle"] = {
        **indep,
        "target_idx": args.oracle_target_idx,
        "ckpt_path": args.oracle_ckpt,
    }
    msg = (f"[eval] independent_mu_median={indep['independent_mu_median']:.4f}  "
           f"mean={indep['independent_mu_mean']:.4f}  "
           f"top{args.top_n}={indep['independent_mu_top_mean']:.4f}")
    if "mu_wasserstein_vs_train" in indep:
        msg += f"  mu_W_vs_train={indep['mu_wasserstein_vs_train']:.4f}"
    print(msg)

    gc = compute_gc_stats(gen_seqs)
    results["gc"] = gc
    print(f"[eval] gc_mean={gc['gc_mean']:.4f}  gc_std={gc['gc_std']:.4f}")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, "eval_metrics.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[eval] saved -> {out_path}")
        # Per-seq oracle scores, so the novelty axis can read them without
        # re-running the Enformer.
        mu_path = os.path.join(args.out_dir, "mu_indep_per_seq.npz")
        payload = {"mu_indep": mu_indep}
        if mu_indep_train is not None:
            payload["mu_indep_train"] = mu_indep_train
        np.savez_compressed(mu_path, **payload)
        print(f"[eval] saved per-seq mu -> {mu_path}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
