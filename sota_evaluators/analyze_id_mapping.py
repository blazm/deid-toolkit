#!/usr/bin/env python3
"""
Identity Mapping Structure Analysis (1:1 / 1:N / N:1 / M:N de-identification).

Characterizes HOW a de-identification technique acts on the SET of identities
(celeba-test), not just on individual images:

  * 1:1  — one person -> one distinct, stable de-identified identity   (consistent)
  * 1:N  — one person -> several different de-identified faces         (identity drift
           / within-person inconsistency)          -> measured by PERSON-SPLIT RATE
  * N:1  — several persons -> the same de-identified face/cluster      (identity
           collapse / mixing)                        -> measured by MIXING RATE,
           cluster purity and effective identity count K vs. P
  * M:N  — both simultaneously

Protocol
  1. Person -> image grouping is derived from the id fields in the celeba-test
     genuine/impostor pair files (no new labels needed).
  2. Same-identity cosine threshold is calibrated from the ALIGNED set
     (EER threshold of aligned-vs-aligned on the standard pair files).
  3. Each technique's de-identified embeddings are clustered with average-linkage
     agglomerative clustering at that threshold.
  4. Metrics per technique:
       - person-split rate:  fraction of persons (>=2 images) whose images span
                             >=2 clusters               [1:N axis]
       - mixing rate:        fraction of clusters (size>=2) containing >=2
                             original persons           [N:1 axis]
       - mean cluster purity: largest-person share of each cluster (size>=2)
       - K / P:              #clusters vs. #persons
       - CMC-1 misassignment: fraction of de-id images whose NEAREST aligned
                             face belongs to a different person
       - within-person similarity of de-id embeddings (consistency proxy)
  5. The aligned set itself is computed as the reference (should be consistent).

Outputs (per run):
  id_mapping_<model>.csv, id_mapping_<model>_results.json
  id_mapping_<model>_2panel.{png,pdf,svg} + _caption.txt   (quadrant + CMC-1)
  id_mapping_<model>.html

Scatter mode (cross-model), mirroring analyze_linkability.py:
  python analyze_id_mapping.py --scatter \
      --results-a id_mapping_swinface_results.json  --label-a SwinFace \
      --results-b id_mapping_transface_results.json --label-b TransFace \
      --output id_mapping_scatter

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
# I/O
# --------------------------------------------------------------------------

def load_embeddings(d):
    d = Path(d)
    if not d.is_dir():
        return {}, []
    emb = {}
    for p in sorted(d.glob("*.npy")):
        emb[p.stem] = np.load(p).reshape(-1).astype(np.float32)
    keys = sorted(emb)
    return emb, keys


def parse_pairs_file(pairs_path):
    """Returns list of (id1, img1, id2, img2)."""
    rows = []
    with open(pairs_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                rows.append((parts[0], os.path.splitext(parts[1])[0],
                             parts[2], os.path.splitext(parts[3])[0]))
    return rows


def build_person_map(pairs_dir, valid_images):
    """id -> set(images); images restricted to those available in the aligned set."""
    persons = {}
    for fn in ("celeba-test_genuine_pairs.txt", "celeba-test_impostor_pairs.txt"):
        for id1, img1, id2, img2 in parse_pairs_file(os.path.join(pairs_dir, fn)):
            if img1 in valid_images:
                persons.setdefault(id1, set()).add(img1)
            if img2 in valid_images:
                persons.setdefault(id2, set()).add(img2)
    return {pid: imgs for pid, imgs in persons.items() if imgs}


def _savefig_safe(fig, path, **kw):
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
    png = base + ".png"
    _savefig_safe(fig, png, dpi=300)
    _savefig_safe(fig, base + ".pdf", format="pdf")
    if svg:
        _savefig_safe(fig, base + ".svg", format="svg")
    return png, base + ".svg"


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_eer_threshold(g_scores, i_scores):
    labels = np.concatenate([np.ones(len(g_scores)), np.zeros(len(i_scores))])
    scores = np.concatenate([g_scores, i_scores])
    order = np.argsort(-scores, kind="mergesort")
    sl = labels[order]
    tp = np.cumsum(sl)
    fp = np.cumsum(1 - sl)
    far = fp / (len(sl) - sl.sum())
    frr = 1.0 - tp / sl.sum()
    idx = int(np.argmin(np.abs(far - frr)))
    return float(scores[order][idx])


def cluster_deid(mat, same_id_cosine_threshold):
    """Average-linkage agglomerative clustering on cosine distance."""
    from sklearn.cluster import AgglomerativeClustering
    dist_thr = max(1e-6, 1.0 - same_id_cosine_threshold)
    model = AgglomerativeClustering(n_clusters=None,
                                    distance_threshold=dist_thr,
                                    metric="cosine", linkage="average")
    return model.fit_predict(mat)


def mapping_metrics(label_by_img, persons, img_to_person, n_persons):
    """label_by_img: {image_stem: cluster_id}; persons: {pid: set(stems)}."""
    # person-split rate (persons with >= 2 images)
    multi = {p: imgs for p, imgs in persons.items() if len(imgs) >= 2}
    split = 0.0
    if multi:
        n_split = sum(1 for imgs in multi.values()
                      if len({label_by_img[i] for i in imgs}) >= 2)
        split = n_split / len(multi)

    # cluster-level metrics (clusters of size >= 2)
    clusters = {}
    for img, c in label_by_img.items():
        clusters.setdefault(c, []).append(img)
    sizes2 = {c: imgs for c, imgs in clusters.items() if len(imgs) >= 2}
    if sizes2:
        purities, mixed = [], 0
        for imgs in sizes2.values():
            pid_counts = {}
            for img in imgs:
                pid = img_to_person.get(img)
                if pid is not None:
                    pid_counts[pid] = pid_counts.get(pid, 0) + 1
            tot = sum(pid_counts.values())
            purities.append((max(pid_counts.values()) / tot) if tot else 1.0)
            if len(pid_counts) >= 2:
                mixed += 1
        purity = float(np.mean(purities))
        mixing = mixed / len(sizes2)
    else:
        purity, mixing = 1.0, 0.0

    return {
        "n_clusters": len(clusters),
        "n_clusters_multi": len(sizes2),
        "n_persons": n_persons,
        "n_persons_multi": len(multi),
        "person_split_rate": split,
        "cluster_purity": purity,
        "mixing_rate": mixing,
    }


# --------------------------------------------------------------------------
# Per-technique analysis
# --------------------------------------------------------------------------

def discover_techniques(embeddings_root):
    dsets = Path(embeddings_root) / "datasets"
    if not dsets.is_dir():
        return []
    out = []
    for td in sorted(dsets.iterdir()):
        ct = td / "celeba-test"
        if ct.is_dir() and list(ct.glob("*.npy")):
            out.append((td.name, str(ct)))
    return out


def analyze_one(name, deid_emb, aligned_emb, aligned_mat, aligned_keys,
                gal_person, persons, thr):
    """deid_emb: {stem: vec} (subset of aligned_keys). persons: {pid: set(stems)}
    restricted to stems present in deid_emb."""
    keys = [k for k in aligned_keys if k in deid_emb]
    mat = np.stack([deid_emb[k] for k in keys])
    i2p = {k: gal_person.get(k) for k in keys}          # stem -> person id
    persons_here = {p: {i for i in imgs if i in set(keys) and i2p.get(i) == p}
                    for p, imgs in persons.items()}
    persons_here = {p: s for p, s in persons_here.items() if s}
    n_persons = len(persons_here)

    labels_arr = cluster_deid(mat, thr)
    label_by_img = {k: int(l) for k, l in zip(keys, labels_arr)}
    mm = mapping_metrics(label_by_img, persons_here, i2p, n_persons)
    # CMC-1 misassignment against the full aligned gallery
    sims = mat @ aligned_mat.T
    nearest = np.argmax(sims, axis=1)
    mis = 0
    counted = 0
    for row, col in zip(range(len(keys)), nearest):
        own = i2p.get(keys[row])
        other = gal_person.get(aligned_keys[col])
        if own is not None and other is not None:
            counted += 1
            mis += int(own != other)
    cmc1_mis = mis / counted if counted else float("nan")

    # within-person mean de-id cosine similarity (consistency proxy)
    ws = []
    for p, imgs in persons_here.items():
        if len(imgs) >= 2:
            vs = np.stack([deid_emb[i] for i in imgs])
            S = vs @ vs.T
            ws.append(float((S.sum() - np.trace(S)) / (len(vs) * (len(vs) - 1))))
    within = float(np.mean(ws)) if ws else float("nan")

    return {"name": name, **mm, "cmc1_misassignment": cmc1_mis,
            "within_person_sim": within, "n_images": len(keys)}


# --------------------------------------------------------------------------
# Analysis driver
# --------------------------------------------------------------------------

def analyze_model(root, pairs_dir, output_prefix):
    root = Path(root)
    model_name = root.name
    print("=" * 70)
    print(f"Identity Mapping Structure — {model_name}")
    print("=" * 70)
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)) or ".", exist_ok=True)

    aligned_emb, aligned_keys = load_embeddings(root / "aligned" / "celeba-test")
    aligned_mat = np.stack([aligned_emb[k] for k in aligned_keys])
    valid = set(aligned_keys)

    # same-identity threshold from aligned (EER threshold)
    g_rows = parse_pairs_file(os.path.join(pairs_dir, "celeba-test_genuine_pairs.txt"))
    i_rows = parse_pairs_file(os.path.join(pairs_dir, "celeba-test_impostor_pairs.txt"))
    g = [float(np.dot(aligned_emb[r[1]], aligned_emb[r[3]])) for r in g_rows
         if r[1] in aligned_emb and r[3] in aligned_emb]
    imp = [float(np.dot(aligned_emb[r[1]], aligned_emb[r[3]])) for r in i_rows
           if r[1] in aligned_emb and r[3] in aligned_emb]
    thr = compute_eer_threshold(np.array(g), np.array(imp))

    persons_all = build_person_map(pairs_dir, valid)
    gal_person = {}
    for pid, imgs in persons_all.items():
        for i in imgs:
            gal_person[i] = pid
    n_persons_all = len(persons_all)
    n_multi_all = sum(1 for imgs in persons_all.values() if len(imgs) >= 2)

    print(f"Aligned gallery: {len(aligned_keys)} faces, "
          f"{n_persons_all} persons ({n_multi_all} with >=2 images)")
    print(f"Same-identity cosine threshold (aligned EER): {thr:.4f}\n")

    rows = []
    # reference: aligned vs. itself
    ref = analyze_one("aligned", aligned_emb, aligned_emb, aligned_mat, aligned_keys,
                      gal_person, persons_all, thr)
    rows.append(ref)
    print(f"  {'aligned (ref.)':<16} K={ref['n_clusters']:>5}  split={ref['person_split_rate']:.3f}  "
          f"mix={ref['mixing_rate']:.3f}  pur={ref['cluster_purity']:.3f}  "
          f"cmc1_mis={ref['cmc1_misassignment']:.3f}  within={ref['within_person_sim']:.3f}")

    for tech, ct_path in discover_techniques(root):
        deid_emb, _ = load_embeddings(ct_path)
        if not deid_emb:
            continue
        r = analyze_one(tech, deid_emb, aligned_emb, aligned_mat, aligned_keys,
                        gal_person, persons_all, thr)
        rows.append(r)
        print(f"  {r['name']:<16} K={r['n_clusters']:>5}  split={r['person_split_rate']:.3f}  "
              f"mix={r['mixing_rate']:.3f}  pur={r['cluster_purity']:.3f}  "
              f"cmc1_mis={r['cmc1_misassignment']:.3f}  within={r['within_person_sim']:.3f}")

    # ---- CSV ----
    csv_path = os.path.splitext(output_prefix)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["technique", "n_images", "n_persons", "n_clusters",
                    "k_over_p", "n_persons_multi", "person_split_rate_1toN",
                    "cluster_mixing_rate_Nto1", "mean_cluster_purity",
                    "cmc1_misassignment_rate", "within_person_sim"])
        for r in sorted(rows, key=lambda x: (0 if x["name"] == "aligned" else 1, x["name"])):
            w.writerow([r["name"], r["n_images"], r["n_persons"], r["n_clusters"],
                        f"{r['n_clusters']/r['n_persons']:.3f}" if r["n_persons"] else "",
                        r["n_persons_multi"], f"{r['person_split_rate']:.4f}",
                        f"{r['mixing_rate']:.4f}", f"{r['cluster_purity']:.4f}",
                        f"{r['cmc1_misassignment']:.4f}",
                        f"{r['within_person_sim']:.4f}"])
    print(f"\nCSV table saved: {csv_path}")

    # ---- JSON ----
    json_path = os.path.splitext(output_prefix)[0] + "_results.json"
    with open(json_path, "w") as f:
        json.dump({"model": model_name, "threshold": thr,
                   "n_gallery": len(aligned_keys), "n_persons": n_persons_all,
                   "rows": {r["name"]: {k: v for k, v in r.items() if k != "name"}
                            for r in rows}}, f, indent=2)
    print(f"JSON saved:      {json_path}")

    # ---- figure + report ----
    plot_2panel(rows, output_prefix + "_2panel", model_name)
    make_html(rows, thr, n_persons_all, json_path, output_prefix)
    print(f"Done. Figures under {os.path.splitext(output_prefix)[0]}_*")


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

TECHNIQUE_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948",
    "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#af7aa1", "#86bcb6",
    "#cdc500", "#d37295", "#fac864", "#a6d854",
]

def color_for(name):
    if name == "aligned":
        return "#000000"
    return TECHNIQUE_COLORS[sum(ord(c) for c in name) % len(TECHNIQUE_COLORS)]


def _cmc1_percent(names, tech):
    return [next(r["cmc1_misassignment"] for r in tech if r["name"] == n) * 100
            for n in names]


def plot_2panel(rows, base, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tech = [r for r in rows if r["name"] != "aligned"]
    ref = next(r for r in rows if r["name"] == "aligned")
    names = [r["name"] for r in sorted(tech, key=lambda x: x["name"])]
    s = {r["name"]: r["person_split_rate"] * 100 for r in tech}
    m = {r["name"]: r["mixing_rate"] * 100 for r in tech}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 7.5))

    # (a) quadrant: 1:N drift (x) vs N:1 collapse (y)
    for n in names:
        ax1.scatter(s[n], m[n], s=110, color=color_for(n), edgecolor="k",
                    linewidth=0.5, zorder=3)
        ax1.annotate(n, (s[n], m[n]), textcoords="offset points",
                     xytext=(7, 5), fontsize=9)
    ax1.scatter(ref["person_split_rate"] * 100, ref["mixing_rate"] * 100,
                marker="o", s=140, facecolor="none", edgecolor="black",
                linewidth=2.0, zorder=4, label="Aligned (ref.)")
    ax1.axhline(50, color="gray", linestyle=":", linewidth=1)
    ax1.axvline(50, color="gray", linestyle=":", linewidth=1)
    ax1.text(25, 97, "1:1 consistent", ha="center", va="top", fontsize=11,
             color="#336633", fontweight="bold")
    ax1.text(75, 97, "M:N mixed", ha="center", va="top", fontsize=11,
             color="#663366", fontweight="bold")
    ax1.text(25, 3, "N:1 collapse", ha="center", va="bottom", fontsize=11,
             color="#993333", fontweight="bold")
    ax1.text(75, 3, "1:N drift", ha="center", va="bottom", fontsize=11,
             color="#996633", fontweight="bold")
    lim = max([max(s.values()), max(m.values()), ref["person_split_rate"] * 100,
               ref["mixing_rate"] * 100]) * 1.15 + 5
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.set_xlabel("Person-split rate (%) — 1:N identity drift", fontsize=14,
                   fontweight="bold")
    ax1.set_ylabel("Cluster mixing rate (%) — N:1 identity collapse", fontsize=14,
                   fontweight="bold")
    ax1.set_title("(a) Identity mapping structure\n(split vs. mixing, cluster threshold from aligned EER)",
                  fontsize=14, fontweight="bold", pad=14)
    ax1.tick_params(axis="both", which="major", labelsize=12, length=6, direction="inout")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower center", fontsize=10, bbox_to_anchor=(0.42, 0.02))

    # (b) CMC-1 misassignment bars
    vals = _cmc1_percent(names, tech)
    ax2.bar(range(len(names)), vals, color=[color_for(n) for n in names],
            edgecolor="white")
    refv = ref["cmc1_misassignment"] * 100
    ax2.axhline(refv, color="black", linestyle="--", linewidth=1.5,
                label=f"Aligned (ref.) = {refv:.2f}%")
    for b, v in zip(ax2.patches, vals):
        ax2.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax2.set_ylabel("CMC-1 misassignment rate (%)", fontsize=14, fontweight="bold")
    ax2.set_ylim(0, max(max(vals), refv) * 1.18 + 2)
    ax2.set_title("(b) CMC-1 cross-identity match\n(nearest aligned face belongs to a different person)",
                  fontsize=14, fontweight="bold", pad=14)
    ax2.tick_params(axis="both", which="major", labelsize=12, length=6, direction="inout")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=10)

    fig.suptitle(f"Identity Mapping Structure — {model_name}",
                 fontsize=18, fontweight="bold", y=1.00)
    fig.tight_layout()

    cap = (f"Identity mapping structure of anonymization techniques under {model_name} "
           f"(celeba-test, {ref['n_images']} images). De-identified embeddings were clustered "
           f"using average-linkage agglomerative clustering at the cosine threshold corresponding "
           f"to the aligned set's verification EER. "
           f"(a) Each point is one technique: x = person-split rate (1:N identity drift — the "
           f"fraction of persons with >=2 images whose de-identified versions fall into >=2 "
           f"clusters), y = cluster mixing rate (N:1 identity collapse — the fraction of "
           f"multi-image clusters containing >=2 different original persons), with quadrant "
           f"labels for the four mapping regimes (1:1 consistent, 1:N drift, N:1 collapse, M:N "
           f"mixed); the open circle is the aligned reference. "
           f"(b) CMC-1 misassignment rate: for each de-identified image, whether the nearest "
           f"aligned gallery face belongs to a different person than its true identity.")
    with open(base + "_caption.txt", "w", encoding="utf-8") as f:
        f.write(cap + "\n")

    png, svg = export_figure(fig, base)
    plt.close(fig)
    print(f"2-panel figure:  {png}")


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

def make_html(rows, thr, n_persons, json_path, output_prefix):
    with open(json_path) as f:
        data = json.load(f)
    model_name = data["model"]

    parts = []
    parts.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Identity Mapping Structure</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1200px; margin: 2em auto; padding: 0 1em; }
h1 { border-bottom: 2px solid #333; padding-bottom: 0.5em; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; }
th { background: #f5f5f5; text-align: center; }
tr.ref { background: #f0f4f8; font-weight: 600; }
svg, img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; }
</style></head><body>
""")
    parts.append(f"<h1>Identity Mapping Structure (1:1 / 1:N / N:1 / M:N) — {model_name}</h1>\n"
                 f"<p>CelebA-test: {n_persons} persons, "
                 f"same-identity cosine threshold {thr:.4f} (aligned EER).</p>\n")
    parts.append("<table><tr><th>Technique</th><th>K (clusters)</th><th>P (persons)</th>"
                 "<th>K/P</th><th>Person-split (1:N)</th><th>Mixing (N:1)</th>"
                 "<th>Cluster purity</th><th>CMC-1 misassignment</th>"
                 "<th>Within-person sim</th></tr>\n")
    for r in sorted(rows, key=lambda x: (0 if x["name"] == "aligned" else 1, x["name"])):
        cls = " class='ref'" if r["name"] == "aligned" else ""
        parts.append(
            f"<tr{cls}><td style='text-align:left'>{r['name']}</td>"
            f"<td>{r['n_clusters']}</td><td>{r['n_persons']}</td>"
            f"<td>{r['n_clusters']/r['n_persons']:.3f}</td>"
            f"<td>{r['person_split_rate']:.4f}</td><td>{r['mixing_rate']:.4f}</td>"
            f"<td>{r['cluster_purity']:.4f}</td><td>{r['cmc1_misassignment']:.4f}</td>"
            f"<td>{r['within_person_sim']:.4f}</td></tr>\n")
    parts.append("</table>\n")

    base = os.path.splitext(output_prefix)[0] + "_2panel"
    try:
        with open(base + ".svg", encoding="utf-8") as f:
            s = f.read()
        if s.lstrip().startswith("<svg"):
            parts.append(s + "\n")
    except Exception:
        parts.append(f'<img src="{os.path.basename(base + ".png")}">\n')
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
# Cross-model scatter
# --------------------------------------------------------------------------

def plot_scatter(path_a, label_a, path_b, label_b, output_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analyze_linkability import compute_spearman

    A = json.load(open(path_a))
    B = json.load(open(path_b))
    names = sorted(set(A["rows"]) & set(B["rows"]) - {"aligned"})
    if not names:
        print("ERROR: no common techniques.")
        sys.exit(1)
    print(f"Cross-probe: {label_a} vs. {label_b} ({len(names)} techniques)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 7.5))
    for ax, key, unit, title in (
        (ax1, "person_split_rate", "Person-split rate (1:N) — %", "(a) 1:N identity drift"),
        (ax2, "mixing_rate", "Cluster mixing rate (N:1) — %", "(b) N:1 identity collapse"),
    ):
        x = np.array([A["rows"][n][key] for n in names]) * 100
        y = np.array([B["rows"][n][key] for n in names]) * 100
        ax.scatter(x, y, s=90, color=[color_for(n) for n in names],
                   edgecolor="k", linewidth=0.5, zorder=3)
        lim = [min(x.min(), y.min()) * 0.95, max(x.max(), y.max()) * 1.1 + 2]
        if lim[0] < 0:
            lim[0] = 0
        ax.plot(lim, lim, "k--", linewidth=1.2, alpha=0.6, label="y = x")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        for n, xi, yi in zip(names, x, y):
            ax.annotate(n, (xi, yi), textcoords="offset points", xytext=(7, 5), fontsize=9)
        rho, p = compute_spearman(x, y)
        ptxt = f"p={p:.2e}" if p < 0.05 else f"p={p:.3f}"
        ax.text(0.03, 0.97, f"Spearman $\\rho$ = {rho:.3f}\n{ptxt}",
                transform=ax.transAxes, va="top", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray"))
        ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
        ax.set_xlabel(f"{label_a} — {unit}", fontsize=14, fontweight="bold")
        ax.set_ylabel(f"{label_b} — {unit}", fontsize=14, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=12, length=6, direction="inout")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Cross-Probe Agreement — Identity Mapping Structure ({label_a} vs. {label_b})",
                 fontsize=17, fontweight="bold", y=0.995)
    fig.tight_layout()
    png, svg = export_figure(fig, output_prefix)
    plt.close(fig)
    cap = (f"Agreement of identity mapping structure metrics between {label_a} and {label_b} "
           f"across {len(names)} anonymization techniques. "
           f"(a) Person-split rate (1:N identity drift); (b) cluster mixing rate (N:1 identity "
           f"collapse). Points on the dashed y=x line show identical mapping behavior under both "
           f"probe models. Spearman rank correlation annotated per panel.")
    with open(output_prefix + "_caption.txt", "w", encoding="utf-8") as f:
        f.write(cap + "\n")
    print(f"Scatter figure:  {png}")


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scatter", action="store_true",
                   help="Cross-model scatter mode (use --results-a/-b)")
    p.add_argument("--embeddings-root", type=str,
                   help="e.g. .../embeddings/SwinFace")
    p.add_argument("--pairs", type=str,
                   default="D:\\dev\\deid-toolkit\\root_dir\\datasets\\pairs")
    p.add_argument("--output", type=str, default="id_mapping")
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
                     args.output if args.output != "id_mapping" else "id_mapping_scatter")
        return

    if not args.embeddings_root:
        p.error("--embeddings-root is required in model mode")
    analyze_model(os.path.abspath(args.embeddings_root),
                  os.path.abspath(args.pairs),
                  os.path.abspath(args.output))


if __name__ == "__main__":
    main()
