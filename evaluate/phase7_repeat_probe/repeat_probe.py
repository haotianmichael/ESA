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


def parse_graph_partners(path, label="graph"):
    """read_id -> set(partner read_ids), from ANY overlap PAF (Neurosamble/Rawsamble).

    Generic: unions BOTH col1 (qname) and col6 (tname) into a per-read set. Works
    whether the input is canonical/de-duped (each pair once) or non-canonical / has
    both (A,B) and (B,A) -- the set membership dedups either way. Streamed line by
    line (never loaded whole into pandas).
    """
    if not os.path.exists(path):
        die(f"--graph_paf not found: {path}")
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
    info(f"{label} graph_paf: {n_lines} overlap records -> {len(partners)} reads with >=1 partner")
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
    # --graph_paf is the generic overlap graph (Neurosamble OR Rawsamble); --neuro_paf
    # is kept as a backward-compat alias (same dest).
    p.add_argument("--graph_paf", "--neuro_paf", dest="graph_paf", required=True,
                   help="overlap-graph PAF (Neurosamble or Rawsamble; canonical or not)")
    p.add_argument("--ref_paf", required=True, help="reads->REF PAF WITH secondaries")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--graph_label", default="neurosamble",
                   help="which graph produced --graph_paf (e.g. neurosamble / rawsamble)")
    p.add_argument("--dataset", default="ecoli", help="dataset tag (e.g. ecoli / yeast)")
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

    # n_loci  = #distinct genomic loci (gap-clustered) -> interspersed/multi-locus repeats.
    # n_aln   = total alignment records (primary+secondary) -> catches BOTH interspersed
    #           AND tandem repeats (e.g. rDNA array), which n_loci can miss.
    mult = {}
    for rid, al in ref_aligns.items():
        pts = [(a["ref_name"], a["ref_mid"]) for a in al]
        n_loci = count_loci(pts, args.locus_gap)
        n_aln = len(al)
        # primary = the tp:A:P alignment (highest mapq if several / none tagged P)
        prims = [a for a in al if a["tp"] == "P"]
        best = max(prims or al, key=lambda a: a["mapq"])
        mult[rid] = {
            "n_loci": n_loci, "is_repeat": int(n_loci >= 2),
            "n_aln": n_aln, "is_multi": int(n_aln >= 2),
            "primary_ref": best["ref_name"], "primary_mid": best["ref_mid"],
            "mapq": best["mapq"], "is_repeat_mapq": int(best["mapq"] < args.mapq_thr),
        }

    with open(os.path.join(args.out_dir, "read_multiplicity.tsv"), "w") as f:
        f.write("read_id\tn_loci\tis_repeat\tn_aln\tis_multi\tprimary_ref\tprimary_mid\t"
                "mapq\tis_repeat_mapq\n")
        for rid, m in mult.items():
            f.write(f"{rid}\t{m['n_loci']}\t{m['is_repeat']}\t{m['n_aln']}\t{m['is_multi']}\t"
                    f"{m['primary_ref']}\t{m['primary_mid']}\t{m['mapq']}\t{m['is_repeat_mapq']}\n")

    # ---- B) partner sets from the overlap graph (both columns) ----------------
    partners = parse_graph_partners(args.graph_paf, args.graph_label)

    # ---- read-id consistency check (fail loudly on large mismatch) -----------
    ref_ids = set(ref_aligns.keys())
    paf_ids = set(partners.keys())
    inter = ref_ids & paf_ids
    frac = len(inter) / max(1, len(paf_ids))
    info(f"read-id overlap: graph reads={len(paf_ids)} ref-aligned reads={len(ref_ids)} "
         f"intersection={len(inter)} ({100*frac:.1f}% of graph reads have a ref alignment)")
    if frac < 0.30:
        die(f"only {100*frac:.1f}% of graph_paf reads have a reads->REF alignment -- read-ids "
            f"likely mismatch between {args.graph_label} graph and reads_to_ref.paf. Refusing to "
            f"produce an empty/garbage join.")

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
            "n_aln": m["n_aln"] if m else -1,
            "is_multi": m["is_multi"] if m else -1,
        }
    info(f"partners without a primary ref alignment dropped: {dropped_total}")

    with open(os.path.join(args.out_dir, "read_features.tsv"), "w") as f:
        f.write("read_id\tdegree\tpartner_locus_dispersion\tn_loci\tis_repeat\t"
                "is_repeat_mapq\tn_aln\tis_multi\n")
        for rid, ft in features.items():
            f.write(f"{rid}\t{ft['degree']}\t{ft['partner_locus_dispersion']}\t{ft['n_loci']}\t"
                    f"{ft['is_repeat']}\t{ft['is_repeat_mapq']}\t{ft['n_aln']}\t{ft['is_multi']}\n")

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
    y_multi = np.array([e["is_multi"] for e in ev], dtype=int)

    def safe_auc(fn, y, s):
        # AUC needs both classes present
        return float(fn(y, s)) if (y.min() == 0 and y.max() == 1) else float("nan")

    # full {label x predictor} grid: 3 labels x {dispersion, degree}
    label_defs = [
        ("is_repeat_nloci", "n_loci>=2", y_loci),
        ("is_repeat_mapq", "mapq<thr", y_mapq),
        ("is_multi_naln", "n_aln>=2", y_multi),
    ]
    predictors = {"dispersion": disp, "degree": deg}
    labels_grid = {}
    for lname, desc, y in label_defs:
        entry = {"desc": desc, "n_positive": int(y.sum()),
                 "positive_frac": float(y.mean()) if len(y) else 0.0}
        for pname, score in predictors.items():
            entry[f"{pname}_roc_auc"] = safe_auc(roc_auc_score, y, score)
            entry[f"{pname}_pr_auc"] = safe_auc(average_precision_score, y, score)
        labels_grid[lname] = entry

    # pairwise label agreement (robustness across repeat definitions)
    def agreement(y1, y2):
        n = len(y1)
        both1 = int(np.sum((y1 == 1) & (y2 == 1)))
        both0 = int(np.sum((y1 == 0) & (y2 == 0)))
        union_pos = int(np.sum((y1 == 1) | (y2 == 1)))
        jacc = both1 / union_pos if union_pos else float("nan")
        return {"agreement_frac": (both1 + both0) / n if n else float("nan"),
                "both_positive": both1, "both_negative": both0,
                "positive_jaccard": jacc,
                "flag": ("LOW positive overlap -- labels capture different repeat notions; "
                         "treat AUC cautiously" if (jacc == jacc and jacc < 0.5)
                         else "labels broadly consistent")}
    ys = {n: y for n, _, y in label_defs}
    pairwise = {}
    names = list(ys)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            pairwise[f"{names[a]}__vs__{names[b]}"] = agreement(ys[names[a]], ys[names[b]])

    metrics = {
        "graph_label": args.graph_label,
        "dataset": args.dataset,
        "graph_paf": os.path.abspath(args.graph_paf),
        "ref_paf": os.path.abspath(args.ref_paf),
        "params": {"locus_gap": args.locus_gap, "mapq_thr": args.mapq_thr},
        "n_reads_graph": len(partners),
        "n_reads_ref_aligned": len(ref_aligns),
        "n_reads_evaluated": len(ev),
        "n_dropped_partners": dropped_total,
        "labels": labels_grid,
        "spearman_dispersion_vs_nloci": {
            "rho": float(spearmanr(disp, nloci).correlation),
            "p": float(spearmanr(disp, nloci).pvalue),
        },
        "label_agreement_pairwise": pairwise,
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- stdout summary table ------------------------------------------------
    print("\n==================== PHASE 7 REPEAT PROBE ====================")
    print(f"dataset={args.dataset}  graph={args.graph_label}")
    print(f"reads evaluated: {len(ev)}   (graph={len(partners)}, ref-aligned={len(ref_aligns)})")
    print(f"dropped partners (no primary ref aln): {dropped_total}   "
          f"locus_gap={args.locus_gap} mapq_thr={args.mapq_thr}")
    print(f"{'label':18s} {'#pos':>8s} {'pos%':>6s} {'disp_ROC':>9s} {'disp_PR':>9s} "
          f"{'deg_ROC':>9s} {'deg_PR':>9s}")
    print("-" * 74)
    for lname, desc, _ in label_defs:
        e = labels_grid[lname]
        print(f"{desc:18s} {e['n_positive']:>8d} {100*e['positive_frac']:>5.1f}% "
              f"{e['dispersion_roc_auc']:>9.4f} {e['dispersion_pr_auc']:>9.4f} "
              f"{e['degree_roc_auc']:>9.4f} {e['degree_pr_auc']:>9.4f}")
    print("-" * 74)
    print(f"Spearman(dispersion, n_loci) = {metrics['spearman_dispersion_vs_nloci']['rho']:.4f}")
    for k, v in pairwise.items():
        print(f"agreement {k}: {100*v['agreement_frac']:.1f}%  posJaccard={v['positive_jaccard']:.3f}"
              f"  -> {v['flag']}")
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
