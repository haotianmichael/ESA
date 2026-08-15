"""
Torch/faiss-free unit tests for the Phase-4 full-scale helpers:
  * overlap_map_full.filter_neighbors (vectorized canonical/self-exclude filter)
  * sanitize_paf.sanitize (miniasm-safe PAF filtering)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluate"))

from overlap_map_full import filter_neighbors  # noqa: E402
from sanitize_paf import sanitize  # noqa: E402


# --------------------------------------------------------------------------- #
# filter_neighbors
# --------------------------------------------------------------------------- #
def _table():
    # 5 window rows over 4 reads; gid i has lexicographic rank i.
    win_gid = np.array([0, 0, 1, 2, 3], dtype=np.int64)
    win_off = np.array([0, 1000, 500, 700, 900], dtype=np.int64)
    name_rank = np.array([0, 1, 2, 3], dtype=np.int64)
    return win_gid, win_off, name_rank


def test_filter_excludes_self_and_invalid_and_keeps_canonical():
    win_gid, win_off, name_rank = _table()
    # Query read gid=0 (rank 0). Neighbor rows: self(0), gid1, gid2, gid3, invalid(-1).
    nid = np.array([0, 2, 3, 4, -1], dtype=np.int64)
    sc = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    qoff = np.zeros(5, dtype=np.int64)
    t_gid, t_off, q_kept, s_kept = filter_neighbors(0, 0, nid, sc, qoff, win_gid, win_off, name_rank)
    assert t_gid.tolist() == [1, 2, 3]         # self (gid0) and -1 dropped
    assert t_off.tolist() == [500, 700, 900]
    assert np.allclose(s_kept, [0.8, 0.7, 0.6])


def test_filter_drops_noncanonical_smaller_names():
    win_gid, win_off, name_rank = _table()
    # Query read gid=2 (rank 2): gid1 (rank1<2) is non-canonical -> drop; gid3 keep.
    nid = np.array([2, 4], dtype=np.int64)      # rows -> gid1, gid3
    sc = np.array([0.8, 0.6], dtype=np.float32)
    qoff = np.zeros(2, dtype=np.int64)
    t_gid, t_off, q_kept, s_kept = filter_neighbors(2, 2, nid, sc, qoff, win_gid, win_off, name_rank)
    assert t_gid.tolist() == [3]
    assert t_off.tolist() == [900]


def test_filter_empty_input():
    win_gid, win_off, name_rank = _table()
    e = np.empty(0, dtype=np.int64)
    t_gid, t_off, q_kept, s_kept = filter_neighbors(0, 0, e, np.empty(0, np.float32), e,
                                                    win_gid, win_off, name_rank)
    assert t_gid.size == 0 and s_kept.size == 0


# --------------------------------------------------------------------------- #
# sanitize_paf
# --------------------------------------------------------------------------- #
def _paf_line(q, tn, qlen=1000, qs=10, qe=900, strand="+", tlen=1000, ts=20, te=880):
    return "\t".join(str(x) for x in
                     [q, qlen, qs, qe, strand, tn, tlen, ts, te, 800, 880, 60]) + "\tmt:f:0.0\n"


def test_sanitize_drops_selfhit_reversed_shortcols_and_missing(tmp_path):
    inp = tmp_path / "in.paf"
    outp = tmp_path / "out.paf"
    lines = [
        _paf_line("A", "B"),                        # good
        _paf_line("A", "A"),                        # self-hit -> drop
        _paf_line("A", "B", qs=900, qe=10),         # reversed coords -> drop
        "A\t1000\t10\t900\t+\tB\n",                 # <12 cols -> drop
        _paf_line("A", "Z"),                        # Z not in fasta -> drop
        _paf_line("B", "A"),                        # good
    ]
    inp.write_text("".join(lines))
    # fasta lengths equal to the PAF qlen -> rescale is identity
    kept, dropped = sanitize(str(inp), str(outp), fasta_lengths={"A": 1000, "B": 1000})
    assert kept == 2 and dropped == 4
    got = [ln.split("\t")[:6] for ln in outp.read_text().splitlines()]
    assert got[0][0] == "A" and got[0][5] == "B"
    assert got[1][0] == "B" and got[1][5] == "A"


def test_sanitize_rescales_lengths_and_coords_to_fasta(tmp_path):
    inp = tmp_path / "in.paf"
    outp = tmp_path / "out.paf"
    # PAF qlen/tlen = 2000 but the real basecalled read is 1000 bases -> halve.
    inp.write_text(_paf_line("A", "B", qlen=2000, qs=100, qe=900,
                             tlen=2000, ts=200, te=1000))
    kept, dropped = sanitize(str(inp), str(outp), fasta_lengths={"A": 1000, "B": 1000})
    assert kept == 1 and dropped == 0
    f = outp.read_text().splitlines()[0].split("\t")
    assert f[1] == "1000" and f[2] == "50" and f[3] == "450"     # q rescaled /2
    assert f[6] == "1000" and f[7] == "100" and f[8] == "500"    # t rescaled /2
