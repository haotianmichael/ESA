"""
Torch-free unit tests for colinear chaining (overlap_chain.chain_anchors).

A hand-built colinear anchor set must collapse into ONE chain of the right
length; a scattered / anti-colinear set must be rejected. Also pins the
threshold knobs (min_num_anchors, min_chaining_score), the band limit (bw_bp),
and the gap cap (max_gap_bp).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluate"))

from overlap_chain import Chain, chain_anchors  # noqa: E402


def _anchors(points):
    """Build the Phase-1 per-pair dict {(q_off, t_off): score} from bp points
    (samples_per_kmer=1 in these tests, so offset == bp)."""
    return {(q, t): 1.0 for (q, t) in points}


# --------------------------------------------------------------------------- #
# Colinear -> one chain
# --------------------------------------------------------------------------- #
def test_colinear_forms_one_chain_of_right_length():
    # q and t increase together (diff == 0): a perfect diagonal of 5 anchors.
    pts = [(0, 1000), (100, 1100), (200, 1200), (300, 1300), (400, 1400)]
    chains = chain_anchors(
        _anchors(pts), samples_per_kmer=1,
        min_num_anchors=5, min_chaining_score=40.0,
    )
    assert len(chains) == 1
    ch = chains[0]
    assert isinstance(ch, Chain)
    assert ch.n_anchors == 5
    assert ch.q_start == 0 and ch.t_start == 1000
    # span extends one window (WIN_SAMPLES//spk = 2000) beyond the last anchor.
    assert ch.q_end == 400 + 2000
    assert ch.t_end == 1400 + 2000
    assert ch.score >= 40.0


# --------------------------------------------------------------------------- #
# Scattered / anti-colinear -> rejected
# --------------------------------------------------------------------------- #
def test_anti_colinear_is_rejected():
    # q ascending but t strictly descending: no monotone-increasing chain exists,
    # so the longest chain is length 1 -> nothing clears min_num_anchors=5.
    pts = [(0, 500), (100, 400), (200, 300), (300, 200), (400, 100)]
    chains = chain_anchors(
        _anchors(pts), samples_per_kmer=1,
        min_num_anchors=5, min_chaining_score=40.0,
    )
    assert chains == []


def test_too_few_anchors_rejected_by_min_num_anchors():
    pts = [(0, 0), (100, 100), (200, 200)]  # colinear but only 3
    assert chain_anchors(_anchors(pts), 1, min_num_anchors=5, min_chaining_score=1.0) == []
    # lowering the anchor floor accepts the short chain
    got = chain_anchors(_anchors(pts), 1, min_num_anchors=3, min_chaining_score=1.0)
    assert len(got) == 1 and got[0].n_anchors == 3


def test_band_limit_breaks_non_colinear_jump():
    # First two are colinear; the third is reachable in q/t but its |dq-dt| jump
    # (1900) exceeds bw_bp=500, so it cannot extend the chain.
    pts = [(0, 0), (100, 100), (200, 2000)]
    chains = chain_anchors(
        _anchors(pts), 1, min_num_anchors=3, min_chaining_score=1.0, bw_bp=500,
    )
    assert chains == []  # best chain is only length 2 < min_num_anchors


def test_max_gap_splits_into_two_chains():
    # Two tight colinear clusters separated by a > max_gap_bp void -> two chains.
    cluster_a = [(0, 0), (100, 100), (200, 200), (300, 300)]
    cluster_b = [(9000, 9000), (9100, 9100), (9200, 9200), (9300, 9300)]
    chains = chain_anchors(
        _anchors(cluster_a + cluster_b), 1,
        min_num_anchors=4, min_chaining_score=1.0, max_gap_bp=2500,
    )
    assert len(chains) == 2
    assert all(ch.n_anchors == 4 for ch in chains)
