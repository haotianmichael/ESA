"""
SquiggleSeek Step 2a — DTW roles, separated.

Two distinct DTW jobs, evaluated separately (do NOT merge into one number):

* ``dtw_rerank_one``  — pick the best CANDIDATE WINDOW among the top-k by global
  DTW cost. Output is a window coord (judge with the window criterion _covers).
* ``dtw_refine_one``  — subsequence-align the query into the reference region and
  read off a base-pair-precise start. Output is a bp point estimate (judge with
  the symmetric _within).

Both compare in the signal domain: query current signal vs the reference's
expected current signal (pore model). Downsample by ``samples_per_kmer`` (one
point per k-mer) for speed + exactness. The refine context is sized to the
query's own bp span (reads range ~212..1285 bp for ``-r 300``); a fixed 300 bp
context truncated long reads and pushed the start downstream.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from dna2vec.signal_dataset import znormalize  # noqa: E402


def _prep_for_dtw(signal: np.ndarray, ds: int) -> np.ndarray:
    x = znormalize(np.asarray(signal, dtype=np.float32)).astype(np.float64)
    if ds > 1:
        n = (x.shape[0] // ds) * ds
        if n > 0:
            x = x[:n].reshape(-1, ds).mean(axis=1)
    return np.ascontiguousarray(x)


def _est_bp(query_signal: np.ndarray, samples_per_kmer: int) -> int:
    return max(1, int(round(len(query_signal) / samples_per_kmer)))


def dtw_rerank_one(
    query_signal: np.ndarray,
    candidate_coords: List[int],
    reference_seq: str,
    pore_model,
    ds: Optional[int] = None,
) -> Tuple[Optional[int], float]:
    """Pick the candidate window (coord) of minimum global DTW cost.

    Each candidate's expected signal is rendered over the query's own bp span so
    a long read is compared to an equally long reference span (not a fixed
    300 bp window). Returns (best_coord, cost)."""
    from dtaidistance import dtw

    if not candidate_coords:
        return None, float("inf")
    spk = pore_model.samples_per_kmer
    if not ds or ds <= 0:
        ds = spk
    est = _est_bp(query_signal, spk)
    q = _prep_for_dtw(query_signal, ds)

    best_coord, best_cost = None, float("inf")
    for c in candidate_coords:
        exp = pore_model.sequence_to_signal(reference_seq[c : c + est])
        s = _prep_for_dtw(exp, ds)
        cost = dtw.distance_fast(q, s, use_pruning=True)
        if cost < best_cost:
            best_cost, best_coord = cost, c
    return best_coord, best_cost


def dtw_refine_one(
    query_signal: np.ndarray,
    candidate_coords: List[int],
    reference_seq: str,
    pore_model,
    ds: Optional[int] = None,
    ctx_margin: int = 90,
) -> Tuple[Optional[int], float, bool]:
    """Subsequence-align the query into the candidate region -> bp start.

    Context is sized to the query's own bp span (+margin) so long reads are not
    truncated. Returns (reported_start_bp, cost, fallback_triggered)."""
    from dtaidistance.subsequence.dtw import subsequence_alignment

    if not candidate_coords:
        return None, float("inf"), True
    spk = pore_model.samples_per_kmer
    if not ds or ds <= 0:
        ds = spk
    est = _est_bp(query_signal, spk)

    lo, hi = min(candidate_coords), max(candidate_coords)
    region_start = max(0, lo - ctx_margin)
    region_end = min(len(reference_seq), hi + est + ctx_margin)
    cap = est * 2 + 2 * ctx_margin
    if region_end - region_start > cap:  # candidates multi-modal: anchor on top-1
        c0 = candidate_coords[0]
        region_start = max(0, c0 - ctx_margin)
        region_end = min(len(reference_seq), c0 + est + ctx_margin)

    ctx_signal = pore_model.sequence_to_signal(reference_seq[region_start:region_end])
    q = _prep_for_dtw(query_signal, ds)
    s = _prep_for_dtw(ctx_signal, ds)
    if s.shape[0] <= q.shape[0]:  # context still shorter than query -> fallback
        return candidate_coords[0], float("inf"), True

    match = subsequence_alignment(q, s).best_match()
    bp_offset = int(round((match.segment[0] * ds) / spk))
    reported = region_start + bp_offset
    return reported, float(getattr(match, "value", 0.0)), False
