#!/usr/bin/env python3
"""
Phase 7 -- repeat / multi-locus probe on EXISTING Neurosamble embeddings (offline).

Question: does the learned-embedding all-vs-all overlap graph already encode genomic
MULTIPLICITY -- i.e. can we tell, for free, whether a read comes from a repeated /
multi-copy region (rRNA operons, IS elements, ...) purely from the genomic spread of
its overlap partners? If so, Neurosamble embeddings could later drive repeat-aware
targeted sequencing. This is a feasibility GATE: no training, no re-encoding, no
FAISS rebuild. It reuses Phase-4 outputs (neurosamble.paf) + a fresh reads->REF
alignment (produced by run_phase7.sh) and writes only under --out_dir.

Predictor:  partner_locus_dispersion(read) = number of distinct genomic loci among
            the PRIMARY reference midpoints of that read's overlap PARTNERS.
Ground truth (two independent labels -- robustness check):
  is_repeat       = (n_loci >= 2)            # read itself aligns to >=2 distinct loci
  is_repeat_mapq  = (primary_mapq < mapq_thr)  # low primary MAPQ = ambiguous placement
Both high AUC => signal robust to minimap2 secondary settings. Large label
disagreement => flag it, don't trust the number yet.

CORRECTNESS (bitten us before):
  * neuro.paf is canonical/de-duped (name(qname) < name(tname)); a read's partner set
    MUST union over BOTH col1 (qname) and col6 (tname).
  * ref midpoint = (ref_start+ref_end)//2 from PAF cols 8,9 (0-based). Never mix query
    and target coords.
  * primary/secondary from the tp:A: tag (P/S).
  * locus clustering is PER READ, bucketed by ref_name first, then single-linkage
    gap-clustered within a contig with gap = --locus_gap.
  * assert read-id consistency across files; fail loudly on large mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def die(msg, code=2):
    print(f"[error] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def info(msg):
    print(f"[phase7] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# PAF parsing
# --------------------------------------------------------------------------- #
def _tp_tag(fields):
    """primary/secondary from the tp:A: tag (P=primary, S=secondary); default P."""
    for t in fields[12:]:
        if t.startswith("tp:A:"):
            return t[5:]
    return "P"


def parse_ref_paf(path):
    """read_id -> list of alignment dicts {ref_name, ref_mid, mapq, tp}.

    Uses ref midpoint = (tstart+tend)//2 (PAF cols 8,9, 0-based). mapq = col 12.
    """
    if not os.path.exists(path):
        die(f"--ref_paf not found: {path}")
    aligns = {}
    n_lines = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 12:
                continue
            try:
                tstart, tend, mapq = int(c[7]), int(c[8]), int(c[11])
            except ValueError:
                continue
            rid, ref_name = c[0], c[5]
            ref_mid = (tstart + tend) // 2
            aligns.setdefault(rid, []).append(
                {"ref_name": ref_name, "ref_mid": ref_mid, "mapq": mapq, "tp": _tp_tag(c)})
            n_lines += 1
    if not aligns:
        die(f"no alignments parsed from {path}")
    info(f"ref_paf: {n_lines} alignment lines over {len(aligns)} reads")
    return aligns


def parse_neuro_partners(path):
    """read_id -> set(partner read_ids), unioning BOTH PAF columns (canonical input).

    neurosamble.paf is de-duped & canonical (qname < tname), so each pair appears
    once; to recover a read's full neighbor set we must add the partner from
    whichever column the read is NOT in -- i.e. union over col1 and col6.
    """
    if not os.path.exists(path):
        die(f"--neuro_paf not found: {path}")
    partners = {}
    n_lines = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 6:
                continue
            q, t = c[0], c[5]
            if q == t:
                continue
            partners.setdefault(q, set()).add(t)   # BOTH directions -- do not drop one
            partners.setdefault(t, set()).add(q)
            n_lines += 1
    if not partners:
        die(f"no overlap pairs parsed from {path}")
    info(f"neuro_paf: {n_lines} overlap records -> {len(partners)} reads with >=1 partner")
    return partners


# --------------------------------------------------------------------------- #
# per-read locus clustering (across contigs; single-linkage gap threshold)
# --------------------------------------------------------------------------- #
def count_loci(points, gap):
    """points = list of (ref_name, ref_mid). Bucket by ref_name, then single-linkage
    gap-cluster midpoints within each contig; return the total #clusters (loci)."""
    if not points:
        return 0
    by_contig = {}
    for ref_name, mid in points:
        by_contig.setdefault(ref_name, []).append(mid)
    n = 0
    for mids in by_contig.values():
        mids.sort()
        n += 1
        for prev, cur in zip(mids, mids[1:]):
            if cur - prev > gap:
                n += 1
    return n


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Phase 7 repeat/multi-locus probe (offline)")
    p.add_argument("--neuro_paf", required=True, help="Phase-4 neurosamble.paf (canonical ava)")
    p.add_argument("--ref_paf", required=True, help="reads->REF PAF WITH secondaries (from run_phase7.sh)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--locus_gap", type=int, default=20000,
                   help="single-linkage gap (bp) to merge alignment midpoints into one locus")
    p.add_argument("--mapq_thr", type=int, default=5,
                   help="primary MAPQ below this => is_repeat_mapq=1 (ambiguous placement)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- A) ground truth from reads->REF alignments --------------------------
    ref_aligns = parse_ref_paf(args.ref_paf)

    # read_id -> (n_loci, is_repeat, primary_ref, primary_mid, primary_mapq, is_repeat_mapq)
    mult = {}
    for rid, al in ref_aligns.items():
        pts = [(a["ref_name"], a["ref_mid"]) for a in al]
        n_loci = count_loci(pts, args.locus_gap)
        # primary = the tp:A:P alignment (highest mapq if several / none tagged P)
        prims = [a for a in al if a["tp"] == "P"]
        best = max(prims or al, key=lambda a: a["mapq"])
        is_repeat = int(n_loci >= 2)
        is_repeat_mapq = int(best["mapq"] < args.mapq_thr)
        mult[rid] = {
            "n_loci": n_loci, "is_repeat": is_repeat,
            "primary_ref": best["ref_name"], "primary_mid": best["ref_mid"],
            "mapq": best["mapq"], "is_repeat_mapq": is_repeat_mapq,
        }

    with open(os.path.join(args.out_dir, "read_multiplicity.tsv"), "w") as f:
        f.write("read_id\tn_loci\tis_repeat\tprimary_ref\tprimary_mid\tmapq\tis_repeat_mapq\n")
        for rid, m in mult.items():
            f.write(f"{rid}\t{m['n_loci']}\t{m['is_repeat']}\t{m['primary_ref']}\t"
                    f"{m['primary_mid']}\t{m['mapq']}\t{m['is_repeat_mapq']}\n")

    # ---- B) partner sets from neuro.paf (both columns) ------------------------
    partners = parse_neuro_partners(args.neuro_paf)

    # ---- read-id consistency check (fail loudly on large mismatch) -----------
    ref_ids = set(ref_aligns.keys())
    paf_ids = set(partners.keys())
    inter = ref_ids & paf_ids
    frac = len(inter) / max(1, len(paf_ids))
    info(f"read-id overlap: neuro_paf reads={len(paf_ids)} ref-aligned reads={len(ref_ids)} "
         f"intersection={len(inter)} ({100*frac:.1f}% of neuro reads have a ref alignment)")
    if frac < 0.30:
        die(f"only {100*frac:.1f}% of neuro_paf reads have a reads->REF alignment -- read-ids "
            f"likely mismatch between neurosamble.paf and reads_to_ref.paf. Refusing to produce "
            f"an empty/garbage join.")

    # primary (ref_name, mid) lookup for dispersion
    primary = {rid: (m["primary_ref"], m["primary_mid"]) for rid, m in mult.items()}

    # ---- C) predictor: partner_locus_dispersion ------------------------------
    features = {}
    dropped_total = 0
    for rid in partners:
        prs = partners[rid]
        degree = len(prs)
        pts = []
        dropped = 0
        for p in prs:
            if p in primary:
                pts.append(primary[p])
            else:
                dropped += 1
        dropped_total += dropped
        dispersion = count_loci(pts, args.locus_gap) if pts else 1
        if dispersion < 1:
            dispersion = 1
        m = mult.get(rid)
        features[rid] = {
            "degree": degree,
            "partner_locus_dispersion": dispersion,
            "n_loci": m["n_loci"] if m else -1,
            "is_repeat": m["is_repeat"] if m else -1,
            "is_repeat_mapq": m["is_repeat_mapq"] if m else -1,
        }
    info(f"partners without a primary ref alignment dropped: {dropped_total}")

    with open(os.path.join(args.out_dir, "read_features.tsv"), "w") as f:
        f.write("read_id\tdegree\tpartner_locus_dispersion\tn_loci\tis_repeat\tis_repeat_mapq\n")
        for rid, ft in features.items():
            f.write(f"{rid}\t{ft['degree']}\t{ft['partner_locus_dispersion']}\t{ft['n_loci']}\t"
                    f"{ft['is_repeat']}\t{ft['is_repeat_mapq']}\n")

    # ---- D) evaluation on reads WITH a primary ref alignment ------------------
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scipy.stats import spearmanr

    ev = [ft for rid, ft in features.items() if ft["is_repeat"] in (0, 1)]
    if not ev:
        die("no reads have both a partner set and a ref alignment -- nothing to evaluate")
    disp = np.array([e["partner_locus_dispersion"] for e in ev], dtype=float)
    deg = np.array([e["degree"] for e in ev], dtype=float)
    nloci = np.array([e["n_loci"] for e in ev], dtype=float)
    y_loci = np.array([e["is_repeat"] for e in ev], dtype=int)
    y_mapq = np.array([e["is_repeat_mapq"] for e in ev], dtype=int)

    def safe_auc(fn, y, s):
        # AUC needs both classes present
        return float(fn(y, s)) if (y.min() == 0 and y.max() == 1) else float("nan")

    metrics = {
        "params": {"locus_gap": args.locus_gap, "mapq_thr": args.mapq_thr},
        "n_reads_neuro": len(partners),
        "n_reads_ref_aligned": len(ref_aligns),
        "n_reads_evaluated": len(ev),
        "n_dropped_partners": dropped_total,
        "label_is_repeat(n_loci>=2)": {
            "n_positive": int(y_loci.sum()),
            "dispersion_roc_auc": safe_auc(roc_auc_score, y_loci, disp),
            "dispersion_pr_auc": safe_auc(average_precision_score, y_loci, disp),
            "degree_roc_auc": safe_auc(roc_auc_score, y_loci, deg),
            "degree_pr_auc": safe_auc(average_precision_score, y_loci, deg),
        },
        "label_is_repeat_mapq(mapq<thr)": {
            "n_positive": int(y_mapq.sum()),
            "dispersion_roc_auc": safe_auc(roc_auc_score, y_mapq, disp),
            "dispersion_pr_auc": safe_auc(average_precision_score, y_mapq, disp),
            "degree_roc_auc": safe_auc(roc_auc_score, y_mapq, deg),
            "degree_pr_auc": safe_auc(average_precision_score, y_mapq, deg),
        },
        "spearman_dispersion_vs_nloci": {
            "rho": float(spearmanr(disp, nloci).correlation),
            "p": float(spearmanr(disp, nloci).pvalue),
        },
    }

    # label agreement (robustness): confusion + overlap %
    both1 = int(np.sum((y_loci == 1) & (y_mapq == 1)))
    both0 = int(np.sum((y_loci == 0) & (y_mapq == 0)))
    only_loci = int(np.sum((y_loci == 1) & (y_mapq == 0)))
    only_mapq = int(np.sum((y_loci == 0) & (y_mapq == 1)))
    agree = (both1 + both0) / len(ev)
    # Jaccard over the positive sets
    union_pos = int(np.sum((y_loci == 1) | (y_mapq == 1)))
    jacc = both1 / union_pos if union_pos else float("nan")
    metrics["label_agreement"] = {
        "confusion": {"both_repeat": both1, "both_unique": both0,
                      "only_nloci_repeat": only_loci, "only_mapq_repeat": only_mapq},
        "agreement_frac": agree, "positive_jaccard": jacc,
        "flag": ("LOW label agreement -- treat AUC cautiously"
                 if (jacc == jacc and jacc < 0.5) else "labels broadly consistent"),
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- stdout summary table ------------------------------------------------
    L = metrics["label_is_repeat(n_loci>=2)"]
    M = metrics["label_is_repeat_mapq(mapq<thr)"]
    print("\n==================== PHASE 7 REPEAT PROBE ====================")
    print(f"reads evaluated: {len(ev)}   (neuro={len(partners)}, ref-aligned={len(ref_aligns)})")
    print(f"dropped partners (no primary ref aln): {dropped_total}")
    print(f"locus_gap={args.locus_gap}  mapq_thr={args.mapq_thr}")
    print(f"{'metric':34s} {'label=n_loci>=2':>18s} {'label=mapq<thr':>16s}")
    print("-" * 70)
    print(f"{'#positive':34s} {L['n_positive']:>18d} {M['n_positive']:>16d}")
    print(f"{'dispersion ROC-AUC':34s} {L['dispersion_roc_auc']:>18.4f} {M['dispersion_roc_auc']:>16.4f}")
    print(f"{'dispersion PR-AUC':34s} {L['dispersion_pr_auc']:>18.4f} {M['dispersion_pr_auc']:>16.4f}")
    print(f"{'degree ROC-AUC (baseline)':34s} {L['degree_roc_auc']:>18.4f} {M['degree_roc_auc']:>16.4f}")
    print(f"{'degree PR-AUC (baseline)':34s} {L['degree_pr_auc']:>18.4f} {M['degree_pr_auc']:>16.4f}")
    print("-" * 70)
    print(f"Spearman(dispersion, n_loci) = {metrics['spearman_dispersion_vs_nloci']['rho']:.4f}")
    print(f"label agreement: {agree*100:.1f}%  positive-Jaccard={jacc:.3f}  -> "
          f"{metrics['label_agreement']['flag']}")
    print("==============================================================\n")

    # ---- E) figures ----------------------------------------------------------
    make_figures(args.out_dir, disp, deg, nloci, y_loci,
                 metrics["spearman_dispersion_vs_nloci"]["rho"])
    info(f"wrote metrics.json, read_multiplicity.tsv, read_features.tsv and 3 PNGs to {args.out_dir}")


def make_figures(out_dir, disp, deg, nloci, y_loci, rho):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    uniq = disp[y_loci == 0]
    rep = disp[y_loci == 1]

    # 1) dispersion histogram, unique vs repeat overlaid (grayscale)
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    hi = int(max(disp.max(), 2))
    bins = np.arange(1, min(hi, 40) + 2) - 0.5
    ax.hist(uniq, bins=bins, color="0.75", edgecolor="0.4", label="unique (n_loci=1)",
            density=True, alpha=0.9)
    ax.hist(rep, bins=bins, color="0.25", edgecolor="black", label="repeat (n_loci>=2)",
            density=True, alpha=0.6)
    ax.set_xlabel("partner_locus_dispersion")
    ax.set_ylabel("density")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_dispersion_hist.png"), dpi=150)
    plt.close(fig)

    # 2) scatter n_loci vs dispersion (grayscale), annotate Spearman
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    jx = nloci + np.random.default_rng(0).uniform(-0.15, 0.15, size=nloci.shape)
    jy = disp + np.random.default_rng(1).uniform(-0.15, 0.15, size=disp.shape)
    ax.scatter(jx, jy, s=3, c="0.35", alpha=0.3, linewidths=0, rasterized=True)
    ax.set_xlabel("n_loci (reads->REF ground truth)")
    ax.set_ylabel("partner_locus_dispersion")
    ax.set_title(f"Spearman rho = {rho:.3f}", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_mult_vs_dispersion.png"), dpi=150)
    plt.close(fig)

    # 3) ROC curves for dispersion and degree on one axes
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    if y_loci.min() == 0 and y_loci.max() == 1:
        for score, style, lab in ((disp, "-", "dispersion"), (deg, "--", "degree")):
            fpr, tpr, _ = roc_curve(y_loci, score)
            ax.plot(fpr, tpr, style, color="black", lw=1.3, label=lab)
    ax.plot([0, 1], [0, 1], ":", color="0.6", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_roc.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
