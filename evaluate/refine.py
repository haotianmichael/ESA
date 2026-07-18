"""
SquiggleSeek Step 2a — subsequence-DTW refinement to a bp-level single mapping.

Global DTW over near-duplicate stride-15 windows can only pick a *window coord*,
whose ~±30 bp blur exceeds the tolerance (it actually hurt: 68% vs 97% top-1).
Instead we use **subsequence DTW**: render the expected signal of the reference
region spanned by the top-k candidates, align the query as a subsequence, and
read off WHERE it best matches — giving a base-pair-precise start position
(``region_start + offset``), not a window index.

Downsampling by ``samples_per_kmer`` (one point per k-mer) makes the alignment
both fast (C-free but small matrices) and exact; empirically it recovers the
true start to ~0 bp. Uses ``dtaidistance`` (no hand-written DTW).
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


def dtw_refine_one(
    query_signal: np.ndarray,
    candidate_coords: List[int],
    reference_seq: str,
    pore_model,
    unit_length: int,
    ds: Optional[int] = None,
    ctx_margin: int = 90,
    max_ctx_bp: int = 1200,
) -> Tuple[Optional[int], float]:
    """Return (reported_start_bp, score=-dtw_cost) via subsequence alignment.

    The reference context spans all top-k candidate windows (+ ``ctx_margin``);
    if that span is wider than ``max_ctx_bp`` (candidates multi-modal), it falls
    back to a context anchored on the top-1 candidate.
    """
    from dtaidistance.subsequence.dtw import subsequence_alignment

    if not candidate_coords:
        return None, float("-inf")

    spk = pore_model.samples_per_kmer
    if not ds or ds <= 0:
        ds = spk

    lo, hi = min(candidate_coords), max(candidate_coords)
    region_start = max(0, lo - ctx_margin)
    region_end = min(len(reference_seq), hi + unit_length + ctx_margin)
    if region_end - region_start > max_ctx_bp:
        c0 = candidate_coords[0]
        region_start = max(0, c0 - ctx_margin)
        region_end = min(len(reference_seq), c0 + unit_length + ctx_margin)

    ctx_signal = pore_model.sequence_to_signal(reference_seq[region_start:region_end])
    q = _prep_for_dtw(query_signal, ds)
    s = _prep_for_dtw(ctx_signal, ds)
    if s.shape[0] <= q.shape[0]:  # context must be longer than the query
        return candidate_coords[0], float("-inf")

    match = subsequence_alignment(q, s).best_match()
    bp_offset = int(round((match.segment[0] * ds) / spk))
    reported = region_start + bp_offset
    cost = float(getattr(match, "value", 0.0))
    return reported, -cost
