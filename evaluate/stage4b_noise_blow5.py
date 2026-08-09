"""STAGE4b Step 4a — write a (optionally test-only) blow5 with additive Gaussian
noise on the RAW signal, so BOTH SquiggleSeek and RawHash2 read the identical
noised input.

Noise model (confirmed with the user):
  * per read, on the stored RAW int16 signal (NOT pA): noise must live in the file
    because RawHash2 reads the file and does its own raw->pA scaling; raw<->pA is a
    linear scale and k*MAD is a relative scale, so both tools see the same relative
    noise after their own normalization.
  * sigma = k * MAD(raw),  MAD = median(|raw - median(raw)|).  Guard: MAD==0
    (flat/degenerate read) -> skip (no noise, no divide-by-zero).
  * NESTED noise: each read's standard-normal draw z is deterministic from a STABLE
    per-read seed (md5 of base_seed:read_id, process-independent), so the SAME z is
    reused at every k and only its amplitude (k*MAD) changes -> the F1-vs-k curve is
    a pure amplitude function, smoother across levels. k=0 -> no noise (lossless copy
    -> the k=0 head-to-head must reproduce Step 3, which also validates the writer).
  * noised = clip(round(raw + k*MAD*z), int16). R9 raw is far from the int16 rail so
    the clip is just a safety net.

Subset: pass --read_ids to keep only those reads (build the test-only blow5 once
with --k 0).

pyslow5 WRITE API is the one piece that cannot be verified here (repo has only read
examples). It is written defensively (core record fields only; no aux columns, which
mapping does not need) and marked UNVERIFIED; the k=0 head-to-head self-check
(SS~91.6 / RawHash2~97.2) is the end-to-end validation. A cheap re-open read-count
check runs at the end to catch gross write failures early.
"""
from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np


def _read_seed(base_seed, rid):
    """Stable (process-independent) per-read seed so nested noise z is identical
    across separate per-k invocations."""
    h = hashlib.md5(f"{base_seed}:{rid}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def parse_args():
    p = argparse.ArgumentParser(description="STAGE4b additive-Gaussian noise blow5 writer")
    p.add_argument("--in_blow5", required=True)
    p.add_argument("--out_blow5", required=True)
    p.add_argument("--k", type=float, required=True, help="noise level: sigma = k*MAD(raw); k=0 = lossless copy")
    p.add_argument("--read_ids", default=None, help="optional file of read_ids to keep (subset)")
    p.add_argument("--base_seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    import pyslow5

    keep = None
    if args.read_ids:
        with open(args.read_ids) as f:
            keep = {ln.strip() for ln in f if ln.strip()}
        print(f"[noise] restricting to {len(keep)} read-ids", flush=True)

    sin = pyslow5.Open(args.in_blow5, "r")
    sout = pyslow5.Open(args.out_blow5, "w")

    # --- header ------------------------------------------------------------------
    # get_empty_header() returns a dict whose values are None; write_header encodes
    # every value, so None -> '<NoneType>.encode' crash. This blow5's run-metadata
    # header attrs are absent anyway (and not needed for mapping — the per-record
    # digitisation/offset/range/sampling_rate are), so coerce every field to a
    # string (empty for the missing ones).
    header = sout.get_empty_header()          # UNVERIFIED pyslow5 write API
    for n in list(header.keys()):
        header[n] = "" if header[n] is None else str(header[n])
    sout.write_header(header)                  # UNVERIFIED

    n_in = n_out = n_noised = n_flat = 0
    stds = []   # (k, mean std-ratio) sample for the spot-check
    for rec in sin.seq_reads(pA=False):        # RAW int16 signal
        n_in += 1
        rid = rec["read_id"]
        if keep is not None and rid not in keep:
            continue
        raw = np.asarray(rec["signal"], dtype=np.float64)
        std0 = float(raw.std())
        if args.k > 0:
            med = float(np.median(raw))
            mad = float(np.median(np.abs(raw - med)))
            if mad > 0:
                z = np.random.default_rng(_read_seed(args.base_seed, rid)).standard_normal(len(raw))
                raw = raw + np.round(args.k * mad * z)
                n_noised += 1
            else:
                n_flat += 1
        out_sig = np.clip(np.round(raw), -32768, 32767).astype(np.int16)
        if len(stds) < 5 and std0 > 0:
            stds.append((rid, std0, float(out_sig.astype(np.float64).std())))

        record = sout.get_empty_record()       # core fields only (no aux)  # UNVERIFIED
        if isinstance(record, tuple):
            record = record[0]
        record["read_id"] = rid
        record["read_group"] = 0
        for fld in ("digitisation", "offset", "range", "sampling_rate"):
            if rec.get(fld) is not None:
                record[fld] = rec[fld]
        record["len_raw_signal"] = int(len(out_sig))
        record["signal"] = out_sig
        sout.write_record(record)              # UNVERIFIED
        n_out += 1

    sout.close()
    sin.close()

    # cheap re-open sanity: read the output back, count reads
    n_check = 0
    try:
        s2 = pyslow5.Open(args.out_blow5, "r")
        for _ in s2.seq_reads(pA=False):
            n_check += 1
        s2.close()
    except Exception as e:                      # noqa: BLE001
        print(f"[noise][ERROR] could not re-open written blow5: {e}", flush=True)
        sys.exit(2)

    print(f"[noise] in={n_in} written={n_out} reopened={n_check} noised={n_noised} "
          f"flat_skipped={n_flat}  k={args.k} base_seed={args.base_seed}", flush=True)
    for rid, s0, s1 in stds:
        print(f"[noise][spotcheck] {rid}: std {s0:.1f} -> {s1:.1f}  (ratio {s1/max(s0,1e-9):.3f})", flush=True)
    if n_out != n_check:
        print(f"[noise][ERROR] written({n_out}) != reopened({n_check}) — writer bug, stop.", flush=True)
        sys.exit(2)
    print(f"[noise] OK -> {args.out_blow5}", flush=True)


if __name__ == "__main__":
    main()
