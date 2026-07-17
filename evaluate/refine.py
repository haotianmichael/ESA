"""
SquiggleSeek Step 2a — DTW refinement of top-k signal candidates.

The FAISS retrieval returns top-k reference coordinate candidates per query.
This collapses them to a single mapping: for each candidate window we render its
expected signal (pore model), DTW-align it to the query signal, and keep the
candidate with the smallest alignment cost. Position = that window's coord,
score = -cost.

DTW uses the C-backed ``dtaidistance`` (no hand-written DTW). Both signals are
z-normalized and mean-downsampled before alignment to control length / speed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# reuse the exact training/inference preprocessing normalization
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from dna2vec.signal_dataset import znormalize  # noqa: E402


def _downsample_mean(x: np.ndarray, factor: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if factor <= 1:
        return np.ascontiguousarray(x)
    n = (x.shape[0] // factor) * factor
    if n == 0:
        return np.ascontiguousarray(x)
    return np.ascontiguousarray(x[:n].reshape(-1, factor).mean(axis=1))


def _prep_for_dtw(signal: np.ndarray, ds: int) -> np.ndarray:
    return _downsample_mean(znormalize(np.asarray(signal, dtype=np.float32)), ds)


def dtw_refine_one(
    query_signal: np.ndarray,
    candidate_coords: List[int],
    reference_seq: str,
    pore_model,
    unit_length: int,
    ds: int = 5,
) -> Tuple[Optional[int], float]:
    """Return (best_coord, score=-min_dtw_cost) over the candidate windows."""
    from dtaidistance import dtw

    if not candidate_coords:
        return None, float("-inf")

    q = _prep_for_dtw(query_signal, ds)
    best_coord, best_cost = None, float("inf")
    for coord in candidate_coords:
        bases = reference_seq[coord : coord + unit_length]
        exp = pore_model.sequence_to_signal(bases)
        c = _prep_for_dtw(exp, ds)
        cost = dtw.distance_fast(q, c, use_pruning=True)
        if cost < best_cost:
            best_cost, best_coord = cost, coord
    return best_coord, -best_cost
