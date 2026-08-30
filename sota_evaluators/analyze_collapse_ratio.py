#!/usr/bin/env python3
"""
Per-identity collapse ratio (CelebA-test, precomputed .npy embeddings).

Definition (matches the manuscript's "identity collapse detection" diagnostic):
  For each person p with >=2 images present in BOTH the aligned set and the
  de-identified set:
      disp(X_p)  = 1 - mean pairwise cosine similarity within person p
      ratio(p)   = disp(deid_p) / disp(aligned_p)
  ratio ~ 1  -> intra-identity geometry preserved
  ratio << 1 -> per-identity compaction (identities internally collapse)
  ratio >> 1 -> per-identity expansion / scattering (1:N drift)

NOTE: this measures intra-identity dispersion change. MERGING different persons
onto a shared region is a separate axis, already covered by the cluster mixing
rate / purity in the id-mapping results.

Interpretation anchors (expected, to be checked against output):
  aligned(ref) = 1.00 by construction
  AIDPro  << 1   (strong within-person consistency on shared identities)
  FAMS    >> 1   (per-image drift, within-person sim ~0.10)
  DP2, LDFA, G2Face  >= 1 (scattering family)
  FALCO, IPFA ~ 1 (displacement-preserving)

Usage:
    conda run -n swinface python analyze_collapse_ratio.py \
        --embeddings-root D:\dev\deid-toolkit\root_dir\embeddings \
        --pairs-dir <pairs> --output <out>
"""

import argparse
import csv
import os
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_id_mapping import load_embeddings, build_person_map, discover_techniques


def within_person_dispersion(vecs):
    """1 - mean pairwise cosine over a list of unit vectors; None if <2."""
    if len(vecs) < 2:
        return None
    m = np.stack(vecs)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    cos = m @ m.T
    n = len(vecs)
    idx = np.triu_indices(n, k=1)
    return float(1.0 - cos[idx].mean())


def person_ratios(deid_emb, aligned_emb, persons):
    ratios = []
    for pid, imgs in persons.items():
    # need >=2 images in each side
        a = [aligned_emb[k] for k in sorted(imgs) if k in aligned_emb]
        d = [deid_emb[k] for k in sorted(imgs) if k in deid_emb]
        da = within_person_dispersion(a)
        dd = within_person_dispersion(d)
        if da is None or dd is None:
            continue
        if da < 1e-6:
            continue
        ratios.append(dd / da)
    return ratios


def _load_dir(dp):
    d = Path(dp)
    out = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.npy")):
        out[p.stem] = np.load(p).reshape(-1).astype(np.float32)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings-root", required=True)
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--output", required=True, help="CSV output path")
    args = ap.parse_args()

    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor
    import time

    model_dirs = []
    for d in sorted(Path(args.embeddings_root).iterdir()):
        if d.is_dir() and (d / "aligned" / "celeba-test").is_dir():
            model_dirs.append(d)
    assert model_dirs, f"no model dirs under {args.embeddings_root}"
    rows = []

    with ProcessPoolExecutor(max_workers=16) as ex:
        for model in (d.name for d in model_dirs):
            t0 = time.time()
            mroot = next(d for d in model_dirs if d.name == model)
            aligned_emb = ex.submit(_load_dir, mroot / "aligned" / "celeba-test").result()
            print(f"{model}: aligned loaded {len(aligned_emb)} in {time.time()-t0:.1f}s", flush=True)
            aligned_keys = sorted(aligned_emb)
            valid = set(aligned_keys)
            persons = build_person_map(args.pairs_dir, valid)

            tech_dirs = {}
            for tech, ct_path in discover_techniques(mroot):
                tech_dirs[tech] = ct_path
            print(f"{model}: loading {len(tech_dirs)} techniques in parallel...", flush=True)
            futs = {tech: ex.submit(_load_dir, p) for tech, p in tech_dirs.items()}
            deids = {t: f.result() for t, f in futs.items()}
            print(f"{model}: all loaded in {time.time()-t0:.1f}s", flush=True)

            def emit(name, deid_emb):
                ratios = person_ratios(deid_emb, aligned_emb, persons)
                if not ratios:
                    print(f"  {name}: no multi-image person overlap, skipping")
                    return
                r = np.array(ratios)
                q1, q3 = np.percentile(r, [25, 75])
                rows.append({
                    "model": model,
                    "technique": name,
                    "n_persons": len(r),
                    "median_ratio": float(np.median(r)),
                    "mean_ratio": float(r.mean()),
                    "q1": float(q1),
                    "q3": float(q3),
                    "frac_lt_1": float((r < 1.0).mean()),
                })

            emit("aligned (reference)", aligned_emb)
            for tech, deid_emb in deids.items():
                if deid_emb:
                    emit(tech, deid_emb)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nCollapse ratio (deid/aligned intra-identity dispersion), CelebA-test -> {args.output}")
    print(f"{'model':<11}{'technique':<18}{'n':>5}{'median':>9}{ 'q1':>8}{ 'q3':>8}{'frac<1':>8}")
    for r in rows:
        print(f"{r['model']:<11}{r['technique']:<18}{r['n_persons']:>5}"
              f"{r['median_ratio']:>9.3f}{r['q1']:>8.3f}{r['q3']:>8.3f}{r['frac_lt_1']:>8.3f}")


if __name__ == "__main__":
    main()
