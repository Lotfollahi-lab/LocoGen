"""JASPAR motif scanning.

Shared by the plausibility axis (motif Gini / hits per sequence) and by the
baseline-construction scripts. Lifted verbatim from the research tree's
`evaluate_sequences` module so that no evaluation axis has to import another.
"""
import numpy as np


def _load_jaspar_pssms(release: str, collection: str, tax_group: str,
                       pseudocount: float, max_motifs: int = None):
    """Return list of (motif_id, biopython PSSM) pairs."""
    from pyjaspar import jaspardb
    jdb = jaspardb(release=release)
    kwargs = {"collection": collection}
    if tax_group:
        kwargs["tax_group"] = [tax_group]
    motifs = jdb.fetch_motifs(**kwargs)
    if max_motifs is not None:
        motifs = motifs[:max_motifs]

    pssms = []
    for m in motifs:
        pwm = m.counts.normalize(pseudocounts=pseudocount)
        pssm = pwm.log_odds()
        pssms.append((m.matrix_id, pssm))
    return pssms


def _pssm_thresholds(pssms, background, fpr: float):
    """Compute per-motif score threshold at the given false-positive rate."""
    thresholds = []
    for _, pssm in pssms:
        dist = pssm.distribution(background=background, precision=10**3)
        thresholds.append(dist.threshold_fpr(fpr))
    return thresholds


def scan_jaspar_motifs(seqs, release: str = "JASPAR2022",
                       collection: str = "CORE",
                       tax_group: str = "vertebrates",
                       fpr: float = 1e-3, pseudocount: float = 0.5,
                       max_motifs: int = None,
                       return_per_seq: bool = False):
    """Scan each sequence on both strands against JASPAR PSSMs.

    Returns (default 3-tuple):
        hits_per_seq : (N,) int — total hit count per sequence (any motif)
        motif_freq   : (M,) float — per-motif hit rate averaged over sequences
        motif_ids    : list[str] of length M

    If `return_per_seq=True`, returns a 4-tuple with the additional:
        per_seq_hits : (N, M) int32 — full sequence × motif hit-count matrix
    """
    from Bio.Seq import Seq

    pssms = _load_jaspar_pssms(release, collection, tax_group,
                               pseudocount, max_motifs)
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    thresholds = _pssm_thresholds(pssms, background, fpr)

    N, M = len(seqs), len(pssms)
    hits_per_seq = np.zeros(N, dtype=np.int64)
    motif_hits = np.zeros(M, dtype=np.int64)
    per_seq_hits = np.zeros((N, M), dtype=np.int32) if return_per_seq else None

    for i, s in enumerate(seqs):
        bseq = Seq(s)
        for j, (_, pssm) in enumerate(pssms):
            # `both=True` scans reverse-complement too — regulatory motifs
            # can bind either strand at enhancer elements.
            count = sum(1 for _ in pssm.search(bseq, threshold=thresholds[j],
                                               both=True))
            hits_per_seq[i] += count
            motif_hits[j] += count
            if return_per_seq:
                per_seq_hits[i, j] = count

    motif_freq = motif_hits.astype(np.float64) / max(N, 1)
    motif_ids = [mid for mid, _ in pssms]
    if return_per_seq:
        return hits_per_seq, motif_freq, motif_ids, per_seq_hits
    return hits_per_seq, motif_freq, motif_ids


def compute_jaspar_metrics(gen_seqs, release: str = "JASPAR2022",
                           collection: str = "CORE",
                           tax_group: str = "vertebrates",
                           fpr: float = 1e-3, max_motifs: int = None):
    gen_hits, _, motif_ids = scan_jaspar_motifs(
        gen_seqs, release=release, collection=collection,
        tax_group=tax_group, fpr=fpr, max_motifs=max_motifs)

    return {
        "motif_hits_per_seq_mean": float(gen_hits.mean()),
        "motif_hits_per_seq_median": float(np.median(gen_hits)),
        "motif_coverage": float((gen_hits > 0).mean()),
        "num_motifs": len(motif_ids),
        "fpr_threshold": fpr,
        "release": release,
        "collection": collection,
    }
