"""Per-identity 1:N re-identification metrics (C028 follow-up).
Standard biometric convention: a query 'hits' at rank k if ANY aligned image
of the query's CelebA identity is in the top-k; rank = first occurrence of
the query's identity among the ranked 2,824-gallery.
Identity map (saved intermediate): aligned_stem_identity_map.csv, built once
from CelebA-HQ-to-CelebA-mapping.txt (Hq idx -> CelebA file) +
identity_CelebA.txt (CelebA file -> person id). All 2,824 stems resolve;
the gallery spans 604 persons.
"""
import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_linkability import discover_techniques, load_embeddings  # noqa: E402

MAP_FILE = r"D:\dev\deid-toolkit\root_dir\datasets\original\celeba-test\aligned_stem_identity_map.csv"


def load_identity_map():
    m = {}
    with open(MAP_FILE, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            q = line.rstrip("\n").split("\t")
            if len(q) == 3 and q[2].strip():
                m[q[0]] = int(q[2])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.embeddings_root)
    idmap = load_identity_map()

    aligned_emb, aligned_keys = load_embeddings(root / "aligned" / "celeba-test")
    aligned_mat = np.stack([aligned_emb[k] for k in aligned_keys])
    gal_idx = {k: i for i, k in enumerate(aligned_keys)}
    N = len(aligned_keys)
    unresolved = [k for k in aligned_keys if k not in idmap]
    if unresolved:
        print(f"WARNING: {len(unresolved)} gallery stems not in identity map (first: {unresolved[:5]})")

    ids = [idmap.get(k, ("unresolved", k)) for k in aligned_keys]
    n_id = len(set(ids))
    cnt = Counter(ids)
    null_r1 = float(sum((cnt[a] / N) for a in ids) / N)
    print(f"Aligned gallery: {N} images, {n_id} identities (largest groups: {cnt.most_common(3)})")
    print(f"Null chance R@1 (per-identity, group-size-weighted) = {null_r1:.6f} (1/{n_id} = {1 / n_id:.6f})\n")

    techniques = discover_techniques(root)
    rows = []
    for tech, ct_path in techniques:
        deid_emb, deid_keys = load_embeddings(ct_path)
        if not deid_emb:
            continue
        common = [k for k in deid_keys if k in gal_idx]
        q_mat = np.stack([deid_emb[k] for k in common])
        gal_sub = aligned_mat[np.array([gal_idx[k] for k in common])]
        sim = q_mat @ gal_sub.T
        qids = [idmap.get(k, ("unresolved", k)) for k in common]
        gal_id_to_cols = {}
        for i, k in enumerate(common):
            gal_id_to_cols.setdefault(idmap.get(k, ("unresolved", k)), []).append(i)
        colset_cache = [set(gal_id_to_cols[qid]) for qid in qids]
        ranks = np.zeros(len(common), dtype=int)
        for i in range(len(common)):
            row = sim[i]
            same_cols = np.asarray(sorted(colset_cache[i]))
            best_same = float(row[same_cols].max())
            ranks[i] = int(np.sum(row > best_same)) + 1
        rows.append((tech, float(np.mean(ranks == 1)), float(np.mean(ranks <= 5)),
                     float(np.median(ranks)), float(np.mean(ranks))))
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["technique", "r1_id", "r5_id", "median_rank_id", "mean_rank_id"])
        for r in rows:
            w.writerow(["%r" % r[0], "%.6f" % r[1], "%.6f" % r[2], "%.1f" % r[3], "%.3f" % r[4]])
    print(f"Wrote {args.out}\n")
    for t, r1, r5, med, mean in sorted(rows, key=lambda x: -x[1]):
        print(f"{t:16s} R1={r1:7.4f}  R5={r5:7.4f}  med={med:8.1f}  mean={mean:8.1f}")
    print(f"\nchance R@1 = {null_r1:.4%}   (approx chance R@5 = {5 * null_r1:.4%})")


if __name__ == "__main__":
    main()
