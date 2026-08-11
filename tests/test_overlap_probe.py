"""
Smoke tests for the Neurosamble Phase-1 probe helpers.

Covers the two pieces that need no torch / GPU: read-window tiling
(``overlap_index.tile_read``) and read-pair truth derivation from a
read->reference PAF (``overlap_probe`` truth logic). A 3-read synthetic case is
enough to pin the geometry and the overlap rule.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluate"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from overlap_index import tile_read  # noqa: E402
from overlap_probe import (  # noqa: E402
    canonical_pair,
    derive_read_pair_truth,
    parse_overlap_truth,
    parse_ref_truth_paf,
)


# --------------------------------------------------------------------------- #
# tile_read
# --------------------------------------------------------------------------- #
def test_tile_read_tiles_full_read_not_just_first_window():
    # 4500 samples, win=2000, stride=1000 -> windows at 0,1000,2000,3000.
    # The last window (3000:4500, len 1500 >= win//2) is kept.
    sig = np.arange(4500, dtype=np.float32)
    tiles = tile_read(sig, win=2000, stride=1000)
    offs = [o for o, _ in tiles]
    assert offs == [0, 1000, 2000, 3000]
    # Every full window is exactly `win` long; the tail is the true remainder.
    assert all(w.shape[0] == 2000 for _, w in tiles[:-1])
    assert tiles[-1][1].shape[0] == 1500


def test_tile_read_keeps_and_drops_partial_by_half_window_rule():
    # Non-overlapping tiling (stride == win) isolates the partial-window rule.
    keep = tile_read(np.zeros(160, dtype=np.float32), win=100, stride=100)
    assert [o for o, _ in keep] == [0, 100]      # tail 100:160 len 60 >= 50 -> kept
    drop = tile_read(np.zeros(140, dtype=np.float32), win=100, stride=100)
    assert [o for o, _ in drop] == [0]           # tail 100:140 len 40 < 50 -> dropped


def test_tile_read_short_read():
    # Read shorter than a window: single partial window iff len >= win//2.
    assert [o for o, _ in tile_read(np.zeros(1500, dtype=np.float32), 2000, 1000)] == [0]
    assert tile_read(np.zeros(900, dtype=np.float32), 2000, 1000) == []


# --------------------------------------------------------------------------- #
# canonical_pair
# --------------------------------------------------------------------------- #
def test_canonical_pair_is_order_independent():
    assert canonical_pair("b", "a") == ("a", "b")
    assert canonical_pair("a", "b") == ("a", "b")


# --------------------------------------------------------------------------- #
# derive_read_pair_truth (3-read synthetic case)
# --------------------------------------------------------------------------- #
def _read_info():
    # A & B overlap by 100 bp on contig1 '+'. C abuts B with a 50 bp gap (no
    # overlap). D is on contig1 but '-' strand (must be excluded, same-strand).
    return {
        "A": ("contig1", "+", 0, 1000),
        "B": ("contig1", "+", 900, 1900),
        "C": ("contig1", "+", 1950, 2950),
        "D": ("contig1", "-", 950, 1950),
    }


def test_derive_read_pair_truth_basic_overlap():
    truth = derive_read_pair_truth(_read_info(), {"A", "B", "C"}, min_overlap_bp=100)
    assert truth == {("A", "B")}


def test_derive_read_pair_truth_respects_min_overlap_bp():
    # Require 101 bp: A/B overlap is exactly 100 -> now no true pair.
    truth = derive_read_pair_truth(_read_info(), {"A", "B", "C"}, min_overlap_bp=101)
    assert truth == set()


def test_derive_read_pair_truth_excludes_minus_strand():
    # D ('-') overlaps A and B in coordinates but must NOT pair (same-strand ava).
    truth = derive_read_pair_truth(_read_info(), {"A", "B", "C", "D"}, min_overlap_bp=100)
    assert truth == {("A", "B")}


def test_derive_read_pair_truth_restricts_to_subsample():
    # B not in the subsample -> its overlaps disappear.
    truth = derive_read_pair_truth(_read_info(), {"A", "C"}, min_overlap_bp=100)
    assert truth == set()


# --------------------------------------------------------------------------- #
# PAF parsing round-trip
# --------------------------------------------------------------------------- #
def test_parse_ref_truth_paf_and_derive(tmp_path):
    paf = tmp_path / "truth.paf"
    # qname qlen qstart qend strand tname tlen tstart tend nmatch alen mapq
    paf.write_text(
        "A\t1000\t0\t1000\t+\tcontig1\t5000\t0\t1000\t1000\t1000\t60\n"
        "B\t1000\t0\t1000\t+\tcontig1\t5000\t900\t1900\t1000\t1000\t60\n"
        "C\t1000\t0\t1000\t+\tcontig1\t5000\t1950\t2950\t1000\t1000\t60\n"
    )
    info = parse_ref_truth_paf(str(paf))
    assert info["A"] == ("contig1", "+", 0, 1000)
    truth = derive_read_pair_truth(info, {"A", "B", "C"}, min_overlap_bp=100)
    assert truth == {("A", "B")}


def test_parse_ref_truth_paf_keeps_longest_span(tmp_path):
    paf = tmp_path / "multi.paf"
    paf.write_text(
        "A\t1000\t0\t100\t+\tc1\t5000\t0\t100\t100\t100\t60\n"       # short
        "A\t1000\t0\t900\t+\tc1\t5000\t500\t1400\t900\t900\t60\n"     # longer -> wins
    )
    info = parse_ref_truth_paf(str(paf))
    assert info["A"] == ("c1", "+", 500, 1400)


def test_parse_overlap_truth_pair_list_and_paf(tmp_path):
    # Bare two-column pair list.
    pl = tmp_path / "pairs.txt"
    pl.write_text("A B\nB C\nC C\n")
    assert parse_overlap_truth(str(pl), {"A", "B", "C"}) == {("A", "B"), ("B", "C")}
    # PAF-shaped: tname is column 6.
    paf = tmp_path / "pairs.paf"
    paf.write_text("A\t1\t0\t1\t+\tB\t1\t0\t1\t1\t1\t60\n")
    assert parse_overlap_truth(str(paf), {"A", "B"}) == {("A", "B")}
