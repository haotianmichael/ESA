"""Score one noise-sweep point and append a row to noise_sweep.csv.

Reuses the head-to-head scorers (builtin locus criterion, PR sweep). For the
given point it records three rows — SquiggleSeek@all, SquiggleSeek@P>=99.9%,
RawHash2@all — plus a one-line trend summary and the truth-invariance evidence
(distinct coords + first 5) that verification gate 1 asks for.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rawhash_compare import (  # noqa: E402
    _read_paf, builtin_pafstats, pr_sweep, recall_at_precision,
)

HEADER = ["timestamp", "method", "work_point", "amp_noise", "dwell_std",
          "tp", "fp", "fn", "precision", "recall", "f1", "n_reads", "ref_bp", "checkpoint"]


def _pct(x):
    return f"{x*100:.1f}" if isinstance(x, (int, float)) else "n/a"


def _diag_criteria(truth_paf, ss_paf, tol_bp=15):
    """Show that retrieval-top1 (bp-strict) and @all (locus-overlap) are DIFFERENT
    criteria on the SAME top-1 coords — not a code-path bug. Uses only the PAFs.

    * overlap: tool interval [ts,te] overlaps truth [ts,te] (what @all/pafstats use)
    * strict : |reported_start - true_start| <= tol_bp (bp-accurate start)
    """
    truth = _read_paf(truth_paf)
    tool = _read_paf(ss_paf)
    n = ov = cov = 0
    examples = []
    for q, (tn, tts, tte, tst, _s) in truth.items():
        n += 1
        if q not in tool:
            continue
        _tn2, rts, rte, _st2, _s2 = tool[q]
        # covers = the pilot's retrieval-top1 criterion _covers(coord, true, unit, tol):
        #   (rts - tol) <= true_start < (rts + span + tol)   [span = rte - rts]
        is_cov = (rts - tol_bp) <= tts < (rte + tol_bp)
        # overlap = the @all / pafstats locus criterion
        is_ov = max(tts, rts) < min(tte, rte)
        cov += is_cov
        ov += is_ov
        if len(examples) < 6:
            examples.append((q, tts, rts, rts - tts, is_cov, is_ov))
    print(f"[diag] {ss_paf}", flush=True)
    print(f"[diag] retrieval accuracy under two criteria on the SAME top-1 coords:", flush=True)
    print(f"[diag]   _covers (+/-{tol_bp}bp window)   : {cov}/{n} = {100*cov/n:.1f}%  "
          f"(== the pilot's retrieval-top1)", flush=True)
    print(f"[diag]   locus overlap (>=1bp)      : {ov}/{n} = {100*ov/n:.1f}%  "
          f"(== @all / pafstats)", flush=True)
    print(f"[diag]   -> the gap is the CRITERION, on the identical coords (not a path bug).", flush=True)
    print(f"[diag]   examples (read, true_start, reported_start, diff, covers?, overlap?):", flush=True)
    for q, tt, rt, df, cv, ovl in examples:
        print(f"[diag]     {q[:26]:<26} true={tt:>9} rep={rt:>9} diff={df:>+7} "
              f"covers={'Y' if cv else 'N'} overlap={'Y' if ovl else 'N'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--ss", required=True, help="squiggleseek.paf")
    ap.add_argument("--rh", default=None, help="rawhash2.paf (optional)")
    ap.add_argument("--amp", required=True)
    ap.add_argument("--dwell", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ref_bp", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--target_p", type=float, default=0.999,
                    help="fallback precision target if no RawHash PAF is given")
    ap.add_argument("--diag", action="store_true",
                    help="print the bp-strict vs locus-overlap retrieval gap + 5 examples")
    ap.add_argument("--tol_bp", type=int, default=15)
    a = ap.parse_args()

    # --- gate 1 evidence: truth coords must be identical across noise points ---
    truth = _read_paf(a.truth)
    coords = sorted(int(v[1]) for v in truth.values())
    n_reads = len(coords)
    print(f"[gate1] amp={a.amp} dwell={a.dwell}: distinct truth coords = "
          f"{len(set(coords))} / {n_reads}; first5 = {coords[:5]}", flush=True)

    # --- SquiggleSeek @all, plus the work-point matched to RawHash's precision ---
    ss_all = builtin_pafstats(a.truth, a.ss)
    sweep = pr_sweep(a.truth, a.ss, n_points=50)   # fine grid for an accurate work point

    rh = None
    if a.rh and os.path.exists(a.rh):
        rh = builtin_pafstats(a.truth, a.rh)

    # BUG-1 fix: match the SAME precision target the head-to-head path uses
    # (rawhash_compare's --match_to = RawHash's precision at THIS point), not a
    # hardcoded 0.999. On the full genome SquiggleSeek's precision ceilings below
    # 99.9%, so a fixed 0.999 target is unreachable and collapses to ~0; matching
    # RawHash's actual precision is the meaningful, path-consistent work point.
    target_p = rh["precision"] if (rh and rh.get("precision") is not None) else a.target_p
    best = recall_at_precision(sweep, target_p)
    if best:
        ss_wp = {"tp": best[1], "fp": best[2], "fn": best[3],
                 "precision": best[4], "recall": best[5], "f1": best[6]}
    else:
        ss_wp = {k: None for k in ("tp", "fp", "fn", "precision", "recall", "f1")}

    rows = [("SquiggleSeek", "@all", ss_all),
            ("SquiggleSeek", f"@P>={target_p*100:.1f}%(=RawHash)", ss_wp)]
    if rh:
        rows.append(("RawHash2", "@all", rh))

    # --- append CSV ---
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        for method, wp, m in rows:
            w.writerow([ts, method, wp, a.amp, a.dwell, m["tp"], m["fp"], m["fn"],
                        m["precision"], m["recall"], m["f1"], n_reads, a.ref_bp, a.checkpoint])

    # --- trend line ---
    ss_f1 = ss_all["f1"]
    rh_f1 = rh["f1"] if rh else None
    d = f"{(ss_f1 - rh_f1)*100:+.1f}" if rh_f1 is not None else "n/a"
    print(f"[trend] amp={str(a.amp):>7} dwell={str(a.dwell):>2} | "
          f"SS_F1={_pct(ss_f1)} SS_R@P{target_p*100:.1f}(=RH)={_pct(ss_wp['recall'])} | "
          f"RH_F1={_pct(rh_f1)} RH_R={_pct(rh['recall']) if rh else 'n/a'} | dF1={d}", flush=True)

    # --- BUG-2 diagnostic: the retrieval-top1 (bp-strict) vs @all (locus-overlap)
    # gap is a CRITERION difference, provable from the PAF alone (no rerun). ---
    if a.diag:
        _diag_criteria(a.truth, a.ss, tol_bp=a.tol_bp)


if __name__ == "__main__":
    main()
