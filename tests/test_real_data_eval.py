"""Torch-free unit tests for the real-data head-to-head helpers.

These cover the pieces that must be correct BEFORE any heavy dependency
(torch / pyslow5 / minimap2) is installed:

  * ``read_fasta_name``      — SquiggleSeek must name the contig exactly as
                               minimap2 / RawHash2 do, or the locus scorer never
                               matches (silent zero recall).
  * ``keep_primary_paf``     — one primary line per read in the minimap2 truth.
  * qname overlap reporting  — the PROMPT §六 sanity check that read-id sets line
                               up between a tool PAF and the truth PAF.

Run: ``pytest tests/test_real_data_eval.py`` (no GPU / external tools needed).
"""
import os
import sys

_EVAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluate")
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

import real_data_eval as rde  # noqa: E402
import rawhash_compare as rc   # noqa: E402


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)
    return str(path)


def test_read_fasta_name_single_record(tmp_path):
    fa = _write(tmp_path / "ref.fasta", ">NC_000913.3 Escherichia coli\nACGTACGT\nGGGG\n")
    assert rde.read_fasta_name(fa) == "NC_000913.3"


def test_read_fasta_name_defaults_to_ref_when_headerless(tmp_path):
    fa = _write(tmp_path / "noheader.fasta", "ACGTACGT\n")
    assert rde.read_fasta_name(fa) == "ref"


def test_read_fasta_name_warns_on_multi_record(tmp_path, capsys):
    fa = _write(tmp_path / "multi.fasta", ">chr1\nACGT\n>chr2\nTTTT\n")
    name = rde.read_fasta_name(fa)
    assert name == "chr1"                       # first contig name is used
    assert "records" in capsys.readouterr().out  # and a warning is printed


def test_keep_primary_paf_keeps_highest_mapq(tmp_path):
    # read r1 has two alignments (mapq 10 and 60); the primary (60) must win.
    raw = _write(
        tmp_path / "raw.paf",
        "r1\t100\t0\t100\t+\tchr\t1000\t200\t300\t90\t100\t10\n"
        "r1\t100\t0\t100\t+\tchr\t1000\t500\t600\t95\t100\t60\n"
        "r2\t100\t0\t100\t-\tchr\t1000\t700\t800\t80\t100\t40\n",
    )
    out = str(tmp_path / "truth.paf")
    n = rde.keep_primary_paf(raw, out)
    assert n == 2                                # one line per read
    rows = {ln.split("\t")[0]: ln.split("\t") for ln in open(out).read().splitlines()}
    assert rows["r1"][7] == "500"               # kept the mapq-60 alignment
    assert rows["r1"][11] == "60"


def test_qname_overlap_full_and_partial(tmp_path):
    truth = _write(tmp_path / "truth.paf",
                   "r1\t10\t0\t10\t+\tchr\t100\t0\t10\t10\t10\t60\n"
                   "r2\t10\t0\t10\t+\tchr\t100\t0\t10\t10\t10\t60\n")
    full = _write(tmp_path / "full.paf",
                  "r1\t10\t0\t10\t+\tchr\t100\t0\t10\t10\t10\t60\n"
                  "r2\t10\t0\t10\t+\tchr\t100\t0\t10\t10\t10\t60\n")
    partial = _write(tmp_path / "partial.paf",
                     "r1\t10\t0\t10\t+\tchr\t100\t0\t10\t10\t10\t60\n"
                     "rX\t10\t0\t10\t+\tchr\t100\t0\t10\t10\t10\t60\n")

    ov_full = rc.qname_overlap(truth, full)
    assert ov_full["matched"] == 2 and ov_full["frac"] == 1.0

    ov_partial = rc.qname_overlap(truth, partial)
    assert ov_partial["matched"] == 1 and ov_partial["frac"] == 0.5


def test_report_qname_overlap_prints_ratio(tmp_path, capsys):
    truth = _write(tmp_path / "truth.paf",
                   "r1\t10\t0\t10\t+\tchr\t100\t0\t10\t10\t10\t60\n"
                   "r2\t10\t0\t10\t+\tchr\t100\t0\t10\t10\t10\t60\n")
    tool = _write(tmp_path / "ss.paf",
                  "r1\t10\t0\t10\t+\tchr\t100\t0\t10\t10\t10\t60\n")
    fracs = rde.report_qname_overlap(truth, {"SquiggleSeek": tool})
    assert fracs["SquiggleSeek"] == 0.5
    out = capsys.readouterr().out
    assert "qname-check" in out and "SquiggleSeek" in out
