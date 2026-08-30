#!/usr/bin/env python3
"""
Cross-Condition Linkability Analysis (de-anonymization / re-identification attack).

While `batch_verify_all.py` measures same-condition verification (technique vs.
technique), this script measures whether a DE-IDENTIFIED face can be linked
back to its ORIGINAL (aligned) identity:

  1. Asymmetric 1:1 linkability
       genuine  = (de-id image, ALIGNED image of the SAME person)   [from genuine pairs]
       impostor = (de-id image, ALIGNED image of a DIFFERENT person) [from impostor pairs]
       -> AUC, EER, Accuracy@EER per technique

     Reference: aligned-vs-aligned (same-condition) run on the same pairs,
     reported as the ceiling ("normal verification" performance).

  2. Open-set 1:N re-identification attack
       probe = each de-identified image, gallery = ALL aligned celeba-test faces
       -> R@1, R@5, median/mean rank of the true original, per-query AUC
          (chance = 1/N)

  3. Cross-probe agreement (scatter mode)
       Spearman rank correlation of technique metrics between SwinFace and
       TransFace (per-run JSON outputs are combined).

Outputs (all under the --output prefix dir / same dir):
  linkability_<model>.csv          per-technique table
  linkability_<model>_results.json machine-readable results
  linkability_<model>_2panel.{png,pdf,svg} + _caption.txt
  linkability_<model>_roc.{png,pdf,svg}
  linkability_<model>_rank_cdf.{png,pdf,svg}
  linkability_<model>.html         standalone HTML report (embedded SVG)

Scatter mode (cross-model figure):
  python analyze_linkability.py --scatter \
      --results-a linkability_swinface_results.json  --label-a SwinFace \
      --results-b linkability_transface_results.json --label-b TransFace \
      --output linkability_scatter

Uses precomputed .npy embeddings only — no model inference required.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def load_embeddings(d):
    """Load all .npy files from d (flat). Returns {stem: vec} and sorted keys."""
    d = Path(d)
    if not d.is_dir():
        return {}, []
    emb = {}
    for p in sorted(d.glob("*.npy")):
        emb[p.stem] = np.load(p).reshape(-1).astype(np.float32)
    keys = sorted(emb)
    return emb, keys


def parse_pairs_file(pairs_path):
    """Format per line: <id1> <image1.jpg> <id2> <image2.jpg> -> [(img1, img2)]"""
    pairs = []
    with open(pairs_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                pairs.append((os.path.splitext(parts[1])[0],
                              os.path.splitext(parts[3])[0]))
    return pairs


def _savefig_safe(fig, path, **kw):
    """savefig with retries (Windows file-lock tolerance)."""
    for attempt in range(3):
        try:
            fig.savefig(path, bbox_inches="tight", **kw)
            return
        except PermissionError:
            if attempt == 2:
                print(f"  WARNING: could not save {path} (file locked?)")
                return
            time.sleep(0.5)


def export_figure(fig, base, svg=True):
    """Save figure as PNG (300 dpi) + PDF + optional SVG under a base path."""
    png = base + ".png"
    pdf = base + ".pdf"
    _savefig_safe(fig, png, dpi=300)
    _savefig_safe(fig, pdf, format="pdf")
    if svg:
        _savefig_safe(fig, base + ".svg", format="svg")
    return png, base + ".svg"


# --------------------------------------------------------------------------
# Metrics (mirrors verify_embeddings.py / batch_verify_all.py)
# --------------------------------------------------------------------------

def compute_roc_auc(g_scores, i_scores):
    labels = np.concatenate([np.ones(len(g_scores)), np.zeros(len(i_scores))])
    scores = np.concatenate([g_scores, i_scores])
    order = np.argsort(-scores, kind="mergesort")
    sl = labels[order]
    tp = np.cumsum(sl)
    fp = np.cumsum(1 - sl)
    tpr = np.concatenate([[0.0], tp / sl.sum()])
    fpr = np.concatenate([[0.0], fp / (len(sl) - sl.sum())])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def compute_eer(g_scores, i_scores):
    labels = np.concatenate([np.ones(len(g_scores)), np.zeros(len(i_scores))])
    scores = np.concatenate([g_scores, i_scores])
    order = np.argsort(-scores, kind="mergesort")
    sl = labels[order]
    tp = np.cumsum(sl)
    fp = np.cumsum(1 - sl)
    far = fp / (len(sl) - sl.sum())
    frr = 1.0 - tp / sl.sum()
    idx = int(np.argmin(np.abs(far - frr)))
    eer = float((far[idx] + frr[idx]) / 2)
    thr = float(scores[order][idx])
    acc = float(np.mean(((scores >= thr).astype(int)) == labels))
    return eer, acc, thr


def compute_1n(sim_matrix, aligned_stems):
    """Open-set 1:N. sim_matrix rows = queries in `aligned_stems` order (de-id),
    cols = aligned gallery in `aligned_stems` order. True match = row==col."""
    n = sim_matrix.shape[0]
    diag = np.diag(sim_matrix)
    other = sim_matrix - np.diag(np.full(n, np.inf))  # mask the true column
    ranks = 1 + (other > diag[:, None]).sum(axis=1)   # 1 = true is top score
    r1 = float(np.mean(ranks == 1))
    r5 = float(np.mean(ranks <= 5))
    # per-query AUC: fraction of impostor scores below the true score (ties = 0.5)
    lower = (other < diag[:, None]).sum(axis=1)
    tie = (other == diag[:, None]).sum(axis=1)
    perq_auc = (lower + 0.5 * tie) / (n - 1)
    return {
        "r1": r1,
        "r5": r5,
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "min_rank": int(ranks.min()),
        "max_rank": int(ranks.max()),
        "auc_1n": float(perq_auc.mean()),
        "ranks": ranks,
    }


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def discover_techniques(embeddings_root):
    """Immediate subdirs of <root>/datasets that contain a celeba-test/*.npy set.
    Skips _reversed variants and non-technique folders automatically.
    Returns [(name, path_to_celeba_test)] sorted: alphabetical."""
    dsets = Path(embeddings_root) / "datasets"
    if not dsets.is_dir():
        return []
    out = []
    for td in sorted(dsets.iterdir()):
        ct = td / "celeba-test"
        npys = list(ct.glob("*.npy")) if ct.is_dir() else []
        if npys:
            out.append((td.name, str(ct)))
    return out


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def analyze_model(embeddings_root, pairs_dir, output_prefix):
    root = Path(embeddings_root)
    model_name = root.name  # SwinFace / TransFace

    print("=" * 70)
    print(f"Linkability Analysis — {model_name}")
    print("=" * 70)
    print(f"Embeddings root: {embeddings_root}")
    print(f"Pairs dir:       {pairs_dir}")
    print(f"Output prefix:   {output_prefix}")
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)) or ".", exist_ok=True)

    # ---- aligned (original) gallery ----
    aligned_emb, aligned_keys = load_embeddings(root / "aligned" / "celeba-test")
    aligned_mat = np.stack([aligned_emb[k] for k in aligned_keys])
    gal_idx = {k: i for i, k in enumerate(aligned_keys)}
    print(f"Aligned gallery: {len(aligned_keys)} faces\n")

    genuine_pairs = parse_pairs_file(os.path.join(pairs_dir, "celeba-test_genuine_pairs.txt"))
    impostor_pairs = parse_pairs_file(os.path.join(pairs_dir, "celeba-test_impostor_pairs.txt"))
    print(f"Pairs: {len(genuine_pairs)} genuine, {len(impostor_pairs)} impostor\n")

    # ---- aligned same-condition reference (ceiling) ----
    g_ref = [float(np.dot(aligned_emb[a1], aligned_emb[a2]))
             for a1, a2 in genuine_pairs
             if a1 in aligned_emb and a2 in aligned_emb]
    i_ref = [float(np.dot(aligned_emb[b1], aligned_emb[b2]))
             for b1, b2 in impostor_pairs
             if b1 in aligned_emb and b2 in aligned_emb]
    fpr_ref, tpr_ref, auc_ref = compute_roc_auc(np.array(g_ref), np.array(i_ref))
    eer_ref, acc_ref, _ = compute_eer(np.array(g_ref), np.array(i_ref))
    print(f"[aligned same-condition ref]  AUC={auc_ref:.4f}  EER={eer_ref:.4f}")

    techniques = discover_techniques(embeddings_root)
    if not techniques:
        print("ERROR: no technique folders found.")
        sys.exit(1)
    print(f"Found {len(techniques)} technique(s): {[t[0] for t in techniques]}\n")

    results = []
    for tech, ct_path in techniques:
        deid_emb, deid_keys = load_embeddings(ct_path)
        if not deid_emb:
            print(f"  {tech}: SKIP (no embeddings)")
            continue
        # 1:1 asymmetric scores
        g_s = []
        for a1, a2 in genuine_pairs:
            if a1 in deid_emb and a2 in aligned_emb:
                g_s.append(float(np.dot(deid_emb[a1], aligned_emb[a2])))
        i_s = []
        for b1, b2 in impostor_pairs:
            if b1 in deid_emb and b2 in aligned_emb:
                i_s.append(float(np.dot(deid_emb[b1], aligned_emb[b2])))
        g_arr, i_arr = np.array(g_s), np.array(i_s)
        if len(g_arr) == 0 or len(i_arr) == 0:
            print(f"  {tech}: SKIP (no scoreable pairs)")
            continue
        fpr, tpr, auc = compute_roc_auc(g_arr, i_arr)
        eer, acc, thr = compute_eer(g_arr, i_arr)

        # 1:N open-set — queries ordered by gallery stem (row i vs col i)
        common = [k for k in deid_keys if k in gal_idx]
        q_mat = np.stack([deid_emb[k] for k in common])
        # gallery restricted to same set for a square, self-consistent metric
        gal_sub = aligned_mat[np.array([gal_idx[k] for k in common])]
        sim = q_mat @ gal_sub.T
        one_n = compute_1n(sim, common)

        results.append({
            "name": tech,
            "auc": auc, "eer": eer, "acc": acc,
            "n_genuine": len(g_arr), "n_impostor": len(i_arr),
            "fpr": fpr, "tpr": tpr,
            **{k: one_n[k] for k in
               ("r1", "r5", "median_rank", "mean_rank", "auc_1n", "ranks")},
        })
        print(f"  {tech:<16} 1:1 AUC={auc:.4f} EER={eer:7.4f}   1:N R@1={one_n['r1']:.4f} "
              f"R@5={one_n['r5']:.4f} medrank={one_n['median_rank']:5.0f}")

    if not results:
        print("ERROR: no results.")
        sys.exit(1)

    # ---- CSV ----
    csv_path = os.path.splitext(output_prefix)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["technique", "n_genuine", "n_impostor",
                    "auc_1to1_linkability", "eer_1to1", "acc_eer",
                    "r1_1toN", "r5_1toN", "median_rank_1toN", "mean_rank_1toN",
                    "auc_1toN_perquery"])
        w.writerow(["aligned(same-condition ref)", len(g_ref), len(i_ref),
                    f"{auc_ref:.4f}", f"{eer_ref:.4f}", f"{acc_ref:.4f}", "", "", "", "", ""])
        for r in sorted(results, key=lambda x: x["name"]):
            w.writerow([r["name"], r["n_genuine"], r["n_impostor"],
                        f"{r['auc']:.4f}", f"{r['eer']:.4f}", f"{r['acc']:.4f}",
                        f"{r['r1']:.4f}", f"{r['r5']:.4f}",
                        f"{r['median_rank']:.0f}", f"{r['mean_rank']:.1f}",
                        f"{r['auc_1n']:.4f}"])
    print(f"\nCSV table saved:  {csv_path}")

    # ---- figures ----
    plot_2panel(results, eer_ref, output_prefix + "_2panel", aligned_n=len(aligned_keys),
                model_name=model_name)
    plot_roc(results, fpr_ref, tpr_ref, auc_ref, output_prefix + "_roc", model_name)
    plot_rank_cdf(results, output_prefix + "_rank_cdf", n_gallery=len(aligned_keys))

    # ---- JSON (for cross-model scatter) ----
    json_path = os.path.splitext(output_prefix)[0] + "_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "model": model_name,
            "gallery_size": len(aligned_keys),
            "aligned_ref": {"auc": auc_ref, "eer": eer_ref, "acc": acc_ref},
            "techniques": {
                r["name"]: {k: r[k] for k in
                            ("auc", "eer", "acc", "n_genuine", "n_impostor",
                             "r1", "r5", "median_rank", "mean_rank", "auc_1n")}
                for r in results
            },
        }, f, indent=2)
    print(f"JSON saved:       {json_path}")

    # ---- HTML report ----
    make_html(results, {"auc": auc_ref, "eer": eer_ref, "acc": acc_ref},
              json_path, csv_path, output_prefix)
    print(f"\nDone. Figures under: {os.path.splitext(output_prefix)[0]}_*.png/.pdf/.svg")


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

# Same palette + hashing as batch_verify_all.py for visual consistency
TECHNIQUE_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948",
    "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#af7aa1", "#86bcb6",
    "#cdc500", "#d37295", "#fac864", "#a6d854",
]

def color_for(name):
    return TECHNIQUE_COLORS[sum(ord(c) for c in name) % len(TECHNIQUE_COLORS)]


def setup_axes(ax, xlabel, ylabel):
    ax.tick_params(axis="both", which="major", labelsize=13, length=6, direction="inout")
    ax.set_xlabel(xlabel, fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)


def plot_2panel(results, eer_ref, base, aligned_n, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = sorted(results, key=lambda r: r["eer"])  # sorted by 1:1 EER (privacy)
    names = [r["name"] for r in rs]
    eers = [r["eer"] * 100 for r in rs]
    r1s = [r["r1"] * 100 for r in rs]
    colors = [color_for(n) for n in names]
    chance_r1 = 100.0 / aligned_n

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 7.5))

    # (a) 1:1 linkability EER
    bars = ax1.bar(range(len(rs)), eers, color=colors, edgecolor="white")
    ax1.axhline(50, color="gray", linestyle=":", linewidth=1.2, alpha=0.7,
                label="Chance (50%)")
    ax1.axhline(eer_ref * 100, color="black", linestyle="--", linewidth=1.6,
                label=f"Aligned same-condition EER = {eer_ref*100:.2f}% (ref.)")
    ax1.set_xticks(range(len(rs)))
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax1.set_ylim(0, max(66.0, max(eers) * 1.12, eer_ref * 100 * 1.15))
    setup_axes(ax1, "Anonymization Technique", "1:1 Linkability EER (%)")
    ax1.set_title("(a) Cross-condition 1:1 verification\n(de-identified probe vs. aligned gallery)",
                  fontsize=14, fontweight="bold", pad=16)
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.95)

    # (b) 1:N re-identification R@1
    order_b = sorted(range(len(rs)), key=lambda i: rs[i]["r1"])  # sorted by R@1
    ax2.bar(range(len(rs)), [r1s[i] for i in order_b], color=[colors[i] for i in order_b],
            edgecolor="white")
    ax2.axhline(chance_r1, color="gray", linestyle=":", linewidth=1.2, alpha=0.7,
                label=f"Chance = 1/{aligned_n} ≈ {chance_r1:.3f}%")
    ax2.set_xticks(range(len(rs)))
    ax2.set_xticklabels([names[i] for i in order_b], rotation=45, ha="right", fontsize=10)
    ax2.set_ylim(0, max(r1s) * 1.18 + 0.5)
    setup_axes(ax2, "Anonymization Technique", "Open-set 1:N re-ID rate R@1 (%)")
    ax2.set_title(f"(b) 1:N re-identification attack\n(de-identified probe vs. {aligned_n}-face aligned gallery)",
                  fontsize=14, fontweight="bold", pad=16)
    ax2.legend(loc="upper right", fontsize=10, framealpha=0.95)

    for b, v in zip(ax1.patches, eers):
        ax1.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=8.5)
    for ax, idxs in ((ax2, order_b),):
        for b, i in zip(ax.patches, idxs):
            v = r1s[i]
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8.5)

    fig.suptitle(f"Cross-Condition Linkability — {model_name}",
                 fontsize=18, fontweight="bold", y=1.00)
    fig.tight_layout()

    # caption
    cap = (f"Cross-condition (linkability) attack analysis with {model_name}. "
           f"(a) 1:1 cross-condition verification EER: probes are de-identified images, "
           f"gallery faces are their original aligned counterparts (genuine pairs) or "
           f"aligned faces of different persons (impostor pairs); the dashed line marks the "
           f"same-condition aligned-vs-aligned EER ({eer_ref*100:.2f}%) as an upper reference and 50% is chance. "
           f"(b) Open-set 1:N re-identification success rate (R@1) when each de-identified query "
           f"is matched against the full {aligned_n}-face aligned gallery; "
           f"chance level is 1/{aligned_n} ≈ {chance_r1:.3f}%. "
           f"Lower EER / R@1 indicate stronger identity unlinking.")
    with open(base + "_caption.txt", "w", encoding="utf-8") as f:
        f.write(cap + "\n")

    png, svg = export_figure(fig, base)
    plt.close(fig)
    print(f"2-panel figure:  {png}")


def plot_roc(results, fpr_ref, tpr_ref, auc_ref, base, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(fpr_ref * 100, tpr_ref * 100, color="#000000", linestyle="--", linewidth=3.0,
            label=f"Aligned same-condition (AUC={auc_ref:.4f})")
    for r in sorted(results, key=lambda x: -x["auc"]):
        ax.plot(r["fpr"] * 100, r["tpr"] * 100,
                color=color_for(r["name"]), linewidth=2.5,
                label=f"{r['name']} (AUC={r['auc']:.4f})")
    ax.plot([0, 100], [0, 100], "k:", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("False Positive Rate (%)", fontsize=16, fontweight="bold")
    ax.set_ylabel("True Positive Rate (%)", fontsize=16, fontweight="bold")
    ax.set_title(f"Cross-Condition Linkability — ROC (de-identified probe vs. aligned gallery)\n{model_name}",
                 fontsize=16, fontweight="bold", pad=16)
    ax.tick_params(axis="both", which="major", labelsize=14, length=6, direction="inout")
    ax.set_xlim(-1, 101)
    ax.set_ylim(-1, 101)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=10, ncol=3, framealpha=0.95)
    fig.tight_layout()
    png, svg = export_figure(fig, base)
    plt.close(fig)
    print(f"ROC figure:      {png}")


def plot_rank_cdf(results, base, n_gallery):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8))
    for r in sorted(results, key=lambda x: x["r1"]):
        ranks_pct = np.asarray(r["ranks"]) / n_gallery * 100.0
        order = np.argsort(ranks_pct)
        y = np.arange(1, len(ranks_pct) + 1) / len(ranks_pct)
        ax.plot(ranks_pct[order] * 0 + ranks_pct[order], y,
                color=color_for(r["name"]), linewidth=2.2,
                label=f"{r['name']} (R@1={r['r1']*100:.1f}%)")
    ax.plot([0.1, 100], [0, 1], "k:", linewidth=0.8, alpha=0.4,
            label="Uniform ranking (no identity information)")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 105)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("True-identity rank (as % of gallery size, log scale)",
                  fontsize=16, fontweight="bold")
    ax.set_ylabel("CDF", fontsize=16, fontweight="bold")
    ax.set_title(f"Distribution of true-identity rank in 1:N search\n(de-identified probe vs. {n_gallery}-face aligned gallery)",
                 fontsize=16, fontweight="bold", pad=16)
    ax.tick_params(axis="both", which="major", labelsize=13, length=6, direction="inout")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="lower right", fontsize=9, ncol=2, framealpha=0.95)
    fig.tight_layout()
    png, svg = export_figure(fig, base)
    plt.close(fig)
    print(f"Rank CDF figure: {png}")


def compute_spearman(x, y):
    """Spearman rho + two-sided p-value (t-approximation). No scipy required."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    mx, my = rx.mean(), ry.mean()
    num = ((rx - mx) * (ry - my)).sum()
    den = np.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum())
    rho = float(num / den) if den > 0 else float("nan")
    from math import sqrt
    if n > 2 and abs(rho) < 1:
        t = rho * sqrt((n - 2) / (1 - rho ** 2))
        # two-sided p via incomplete beta — use large-t normal fallback for |t|>3
        try:
            from scipy import stats
            pval = float(2 * stats.t.sf(abs(t), n - 2))
        except Exception:
            pval = float(2 * (1 - 0.5 * (1 + np.sign(t) * _erf(abs(t) / np.sqrt(2)))))
    else:
        pval = 1.0 if abs(rho) >= 1 else 0.0
    return rho, pval


def _erf(x):
    # Abramowitz–Stegun 7.1.26
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return sign * y


def plot_scatter(results_a_path, label_a, results_b_path, label_b, output_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(results_a_path) as f:
        A = json.load(f)
    with open(results_b_path) as f:
        B = json.load(f)

    names = sorted(set(A["techniques"]) & set(B["techniques"]))
    if not names:
        print("ERROR: no common techniques between the two result files.")
        sys.exit(1)
    print(f"Cross-probe agreement: {label_a} vs. {label_b} ({len(names)} common techniques)")

    def col(res, key):
        return [res["techniques"][n][key] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 7.5))
    for ax, key, unit, title in (
        (ax1, "eer", "1:1 Linkability EER (%)", "(a) 1:1 linkability EER"),
        (ax2, "r1", "Open-set 1:N R@1 (%)", "(b) 1:N re-identification R@1"),
    ):
        x = np.asarray(col(A, key)) * 100
        y = np.asarray(col(B, key)) * 100
        sc = ax.scatter(x, y, s=90, zorder=3,
                        color=[color_for(n) for n in names], edgecolor="k", linewidth=0.5)
        lim = [min(x.min(), y.min()) * 0.95, max(x.max(), y.max()) * 1.08]
        if lim[0] < 0:
            lim[0] = 0
        ax.plot(lim, lim, "k--", linewidth=1.2, alpha=0.6, label="y = x")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        for n, xi, yi in zip(names, x, y):
            ax.annotate(n, (xi, yi), textcoords="offset points",
                        xytext=(7, 5), fontsize=9)
        rho, p = compute_spearman(x.astype(float), y.astype(float))
        ptxt = f"p={p:.2e}" if p < 0.05 else f"p={p:.3f}"
        ax.text(0.03, 0.97, f"Spearman $\\rho$ = {rho:.3f}\n{ptxt}",
                transform=ax.transAxes, va="top", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray"))
        ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
        ax.set_xlabel(f"{label_a} — {unit}", fontsize=14, fontweight="bold")
        ax.set_ylabel(f"{label_b} — {unit}", fontsize=14, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=12, length=6, direction="inout")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Cross-Probe Agreement of Cross-Condition Linkability — {label_a} vs. {label_b}",
                 fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout()
    png, svg = export_figure(fig, output_prefix)
    plt.close(fig)

    cap = (f"Cross-probe agreement of cross-condition linkability metrics between {label_a} and "
           f"{label_b} over {len(names)} anonymization techniques. Each marker is one technique. "
           f"(a) 1:1 cross-condition verification EER; (b) open-set 1:N re-identification rate R@1. "
           f"Points on the dashed y=x line indicate identical resistance to the linkability attack "
           f"under both probe models. Spearman rank correlation with p-value is annotated on each panel.")
    with open(output_prefix + "_caption.txt", "w", encoding="utf-8") as f:
        f.write(cap + "\n")
    print(f"Scatter figure:  {png}")


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

def make_html(results, ref, json_path, csv_path, output_prefix):
    model_name = Path(json_path).parent.name if Path(json_path).suffix else ""
    with open(json_path) as f:
        data = json.load(f)
    model_name = data["model"]
    n_gal = data["gallery_size"]

    parts = []
    parts.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Linkability Results</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1200px; margin: 2em auto; padding: 0 1em; }
h1 { border-bottom: 2px solid #333; padding-bottom: 0.5em; }
h2 { margin-top: 1.6em; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; }
th { background: #f5f5f5; text-align: center; }
tr.ref { background: #f0f4f8; font-weight: 600; }
svg, img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; }
</style></head><body>
""")
    parts.append(f"<h1>Cross-Condition Linkability — {model_name}</h1>\n")
    parts.append(f"<p>De-identified probes matched against <b>original aligned</b> "
                 f"celeba-test faces. Gallery size: {n_gal}. Chance 1:N = 1/{n_gal} "
                 f"≈ {100.0/n_gal:.4f}%. 1:1 pairs derived from the standard "
                 f"celeba-test genuine/impostor pair files.</p>\n")

    parts.append("<h2>Results table</h2>\n"
                 "<table><tr><th>Technique</th><th>1:1 AUC</th><th>1:1 EER</th><th>Acc@EER</th>"
                 "<th>R@1 (1:N)</th><th>R@5 (1:N)</th><th>Median rank</th><th>Mean rank</th>"
                 "<th>1:N per-query AUC</th></tr>\n")
    parts.append(f"<tr class='ref'><td>Aligned (same-condition ref.)</td>"
                 f"<td>{ref['auc']:.4f}</td><td>{ref['eer']:.4f}</td><td>{ref['acc']:.4f}</td>"
                 f"<td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr>\n")
    for r in sorted(results, key=lambda x: x["r1"]):
        parts.append(
            f"<tr><td style='text-align:left'>{r['name']}</td>"
            f"<td>{r['auc']:.4f}</td><td>{r['eer']:.4f}</td><td>{r['acc']:.4f}</td>"
            f"<td>{r['r1']:.4f}</td><td>{r['r5']:.4f}</td>"
            f"<td>{r['median_rank']:.0f}</td><td>{r['mean_rank']:.1f}</td>"
            f"<td>{r['auc_1n']:.4f}</td></tr>\n")
    parts.append("</table>\n")

    for tag in ("_2panel", "_roc", "_rank_cdf"):
        fig_key = "linkability"
        base = os.path.splitext(output_prefix)[0] + tag
        svg = base + ".svg"
        png = base + ".png"
        try:
            with open(svg, encoding="utf-8") as f:
                s = f.read()
            if s.lstrip().startswith("<svg"):
                parts.append(s + "\n")
            else:
                parts.append(f'<img src="{os.path.basename(png)}">\n')
        except Exception:
            parts.append(f'<img src="{os.path.basename(png)}">\n')
        cap_file = base + "_caption.txt"
        if os.path.exists(cap_file):
            with open(cap_file, encoding="utf-8") as f:
                parts.append(f"<p style='color:#444'>{f.read().strip()}</p>\n")

    parts.append("</body></html>")
    out = os.path.splitext(output_prefix)[0] + ".html"
    with open(out, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"HTML report:     {out}")


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--scatter", action="store_true",
                      help="Cross-model scatter mode (use --results-a/-b)")
    mode.add_argument("--model", action="store_true",
                      help="(default) Run linkability analysis for one model root")

    # model mode
    p.add_argument("--embeddings-root", type=str,
                   help="e.g. .../embeddings/SwinFace")
    p.add_argument("--pairs", type=str,
                   default="D:\\dev\\deid-toolkit\\root_dir\\datasets\\pairs",
                   help="Pairs directory with celeba-test pairs")
    p.add_argument("--output", type=str, default="linkability",
                   help="Output prefix, e.g. linkability_swinface")

    # scatter mode
    p.add_argument("--results-a", type=str)
    p.add_argument("--label-a", type=str, default="Model A")
    p.add_argument("--results-b", type=str)
    p.add_argument("--label-b", type=str, default="Model B")

    args = p.parse_args()

    if args.scatter:
        for a in (args.results_a, args.results_b):
            if not a or not os.path.exists(a):
                print(f"ERROR: results file not found: {a}")
                sys.exit(1)
        plot_scatter(args.results_a, args.label_a,
                     args.results_b, args.label_b,
                     args.output if args.output != "linkability" else "linkability_scatter")
        return

    if not args.embeddings_root:
        p.error("--embeddings-root is required in model mode")

    analyze_model(os.path.abspath(args.embeddings_root),
                  os.path.abspath(args.pairs),
                  os.path.abspath(args.output))


if __name__ == "__main__":
    main()
