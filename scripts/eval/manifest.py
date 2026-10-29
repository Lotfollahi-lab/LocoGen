"""Roster of LocoGen (cell, seed) eval invocations behind the reported 3-axis numbers.

Single source of truth for the seed-aggregation step
(`scripts.eval.aggregate_seeds`), so the runs behind the reported LocoGen
numbers are inspectable and re-runnable. The LSF dispatcher and the per-cell
table assembly live in the authors' research tree and are not part of this
repository.

This covers LocoGen at the reference setting (λ=0.7, r=3.0) over 3 cell lines
× 3 seeds. The baselines it is compared against (MDLM, SVDD, DRAKES,
DNA-Diffusion) are third-party methods whose samplers are not in this
repository, so their sequences cannot be regenerated here and are not
enumerated. Sweeps and ablations are likewise not enumerated.

Any other sequence CSV — a baseline you generated yourself, a sweep point, an
ablation — can be scored directly, since every measurement script takes an
explicit path:

    python -m scripts.eval.property.pred_activity      --generated <csv> ...
    python -m scripts.eval.property.atac_accessibility --label L --csv <csv> ...
    python -m scripts.eval.novelty.authenticity_rarity --generated <csv> ...
    python -m scripts.eval.plausibility.motif_gini_hits --label L --csv <csv> ...

Each entry yields the arguments needed for one evaluation:

    {label, cell, seed, generated_csv, train_csv, target_idx, out_dir}

`train_csv` is the per-cell training subsample used as the reference
distribution (`pred_activity --train_data`), not an evaluated method.

K562 and SK-N-SH have a cell-conditional separator + oracle, so they use their
own sample dir when one exists; otherwise the HepG2-anchored samples are scored
under that cell's oracle head.

CLI helpers (`__main__`):
    python -m scripts.eval.manifest --count
    python -m scripts.eval.manifest --index 5    # prints KEY=val lines
    python -m scripts.eval.manifest --list
"""
import argparse
import os

PROJECT_ROOT = os.environ.get(
    "LOCOGEN_PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

ORACLE_HEAD = {"hepg2": 0, "k562": 1, "sknsh": 2}
ALL_CELLS = ("hepg2", "k562", "sknsh")
ALL_SEEDS = (0, 1, 2)

# LocoGen reference setting: λ=0.7, alpha_ratio=3.0, ood_sil_localized=true,
# ood_sil_threshold=0.5, sil_t_threshold=0.3, alpha_cap=20.
LOCOGEN_LABEL = "LocoGen (λ=0.7)"
LOCOGEN_NAME = "v6_d_lam0.7_tau0.3"
LOCOGEN_LAM = 0.7
LOCOGEN_R = 3.0


def _eval_dir(generated_csv: str, cell: str) -> str:
    """Place the eval JSON next to the seqs CSV.

    Cell-aware samples get `<sample_dir>/eval/` — the cell is already encoded
    in the sample-dir name. HepG2-anchored samples scored under the K562 or
    SK-N-SH head get `<sample_dir>/eval_{cell}/`, so the HepG2 result
    (= `eval/`) and the cross-eval JSONs coexist under the same sample dir.
    """
    parent = os.path.dirname(generated_csv)
    if f"_{cell}_seed" in generated_csv:
        return os.path.join(parent, "eval")
    return os.path.join(parent, f"eval_{cell}")


def _abs(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(PROJECT_ROOT, rel)


def _per_cell_seed_entries(cell: str, seed: int):
    """Method × (cell, seed) entries to evaluate.

    LocoGen uses the cell-aware sample dir when one exists (K562/SK-N-SH have
    a cell-conditional separator + oracle); otherwise it falls back to the
    HepG2-anchored samples scored under this cell's oracle head.
    """
    s = f"_{cell}_seed{seed}"
    cell_aware_rel = (
        f"runs/mdlm_mood/lam{LOCOGEN_LAM}_r{LOCOGEN_R}_ms0_{LOCOGEN_NAME}{s}/sequences.csv"
    )
    if cell != "hepg2" and os.path.isfile(_abs(cell_aware_rel)):
        rel = cell_aware_rel
    else:
        rel = (f"runs/mdlm_mood/lam{LOCOGEN_LAM}_r{LOCOGEN_R}_ms0_"
               f"{LOCOGEN_NAME}_hepg2_seed{seed}/sequences.csv")
    return [(LOCOGEN_LABEL, rel)]


def build_work():
    """Return the headline eval work list as a list of dicts (deterministic order)."""
    work = []
    for cell in ALL_CELLS:
        target_idx = ORACLE_HEAD[cell]
        train_csv = _abs(f"runs/train_subsample_{cell}_5k.csv")
        for seed in ALL_SEEDS:
            for label, rel in _per_cell_seed_entries(cell, seed):
                generated_csv = _abs(rel)
                out_dir = _eval_dir(generated_csv, cell)
                work.append({
                    "label": label,
                    "cell": cell,
                    "seed": seed,
                    "target_idx": target_idx,
                    "train_csv": train_csv,
                    "generated_csv": generated_csv,
                    "out_dir": out_dir,
                })
    return work


def _print_index(idx: int):
    work = build_work()
    if idx < 1 or idx > len(work):
        raise SystemExit(f"index {idx} out of range [1, {len(work)}]")
    w = work[idx - 1]
    for k in ("label", "cell", "seed", "target_idx", "train_csv",
              "generated_csv", "out_dir"):
        v = w[k]
        # quote labels (may contain spaces) for safe `eval $(...)` consumption
        if isinstance(v, str) and (" " in v or "(" in v):
            print(f'{k.upper()}="{v}"')
        else:
            print(f'{k.upper()}={v}')


def _print_count():
    print(len(build_work()))


def _print_all():
    for i, w in enumerate(build_work(), 1):
        print(f"[{i:3d}] {w['cell']:5s} seed{w['seed']} "
              f"target={w['target_idx']}  {w['label']:30s}  "
              f"-> {w['generated_csv']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", action="store_true")
    g.add_argument("--index", type=int)
    g.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.count:
        _print_count()
    elif args.index is not None:
        _print_index(args.index)
    else:
        _print_all()
