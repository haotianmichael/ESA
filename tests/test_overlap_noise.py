"""
Torch-free unit tests for the Phase-3 event-noise sweep helpers:
  * event_residual_ratio (intrinsic-noise calibration math)
  * n50 / aun (contiguity from unitig lengths)
  * parse_chained / parse_pafstats (log parsing)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluate"))

from neurosamble_noise_calib import event_residual_ratio  # noqa: E402
from neurosamble_noise_aggregate import (  # noqa: E402
    aun,
    n50,
    parse_chained,
    parse_pafstats,
)


# --------------------------------------------------------------------------- #
# event_residual_ratio
# --------------------------------------------------------------------------- #
def test_residual_ratio_zero_for_pure_block_constant_signal():
    # A perfectly block-constant signal (block=5) has no within-block residual.
    sig = np.repeat(np.arange(20, dtype=np.float64) * 100.0, 5)  # 20 blocks of len 5
    assert event_residual_ratio(sig, block=5) == 0.0


def test_residual_ratio_flat_read_is_nan():
    assert np.isnan(event_residual_ratio(np.zeros(50), block=9))


def test_residual_ratio_between_0_and_1_for_noisy_signal():
    rng = np.random.default_rng(0)
    base = np.repeat(rng.normal(0, 500, 100), 9)          # block-constant baseline
    noisy = base + rng.normal(0, 50, base.shape)          # small within-block noise
    r = event_residual_ratio(noisy, block=9)
    assert 0.0 < r < 1.0


# --------------------------------------------------------------------------- #
# n50 / aun
# --------------------------------------------------------------------------- #
def test_n50_basic():
    # lengths 1..6 sum=21, half=10.5; sorted desc 6(acc6),5(acc11>=10.5) -> N50=5
    assert n50([1, 2, 3, 4, 5, 6]) == 5
    assert n50([]) == 0


def test_aun_basic():
    # (2*2 + 4*4) / (2+4) = 20/6
    assert abs(aun([2, 4]) - 20.0 / 6.0) < 1e-9
    assert aun([]) == 0.0


# --------------------------------------------------------------------------- #
# log parsing
# --------------------------------------------------------------------------- #
def test_parse_pafstats(tmp_path):
    p = tmp_path / "pafstats_neurosamble.err"
    p.write_text(
        "TP: 16538, FP: 111, FN: 5491, TN: 0\n"
        "Precision: 0.9933329329088835\n"
        "Recall: 0.7507376639883789\n"
        "F1 Score: 0.8551631418377372\n"
        "Traceback (most recent call last):\n"
        "ZeroDivisionError: float division by zero\n"
    )
    got = parse_pafstats(str(p))
    assert abs(got["recall"] - 0.750737) < 1e-5
    assert abs(got["f1"] - 0.855163) < 1e-5


def test_parse_chained(tmp_path):
    p = tmp_path / "chained_reads.out"
    p.write_text(
        "---- neurosamble ----\n"
        "unitig\tchr\tn\tk\tR\n"
        "utg000001l\tAE014075.1\t5\t3\t0.600\n"
        "TOTAL\t-\t42\t26\t0.619\n"
        "---- rawsamble ----\n"
        "unitig\tchr\tn\tk\tR\n"
        "TOTAL\t-\t39\t19\t0.487\n"
    )
    assert abs(parse_chained(str(p), "neurosamble") - 0.619) < 1e-9
    assert abs(parse_chained(str(p), "rawsamble") - 0.487) < 1e-9
    assert np.isnan(parse_chained(str(p), "mm2"))
