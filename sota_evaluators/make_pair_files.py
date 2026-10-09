#!/usr/bin/env python3
"""
Generate genuine/impostor pair files for datasets missing them (kdef, rafd).

Stricter than the legacy generator (deid/utils/generate_img_pairs_all.py):

  * genuine pairs: ALL combinations of 2 DISTINCT images of the same
    identity (duplicates removed; a pair line always has img1 != img2)
  * impostor pairs: same count as genuine, sampled between DIFFERENT
    identities only, with NO duplicated pair rows (a (imgA, imgB) pair,
    in either order, is emitted at most once)
  * only images that actually exist in datasets/aligned/<ds> are used
  * id fields are written from the CURRENT label file (no stale ids)
  * seed 42, matching the legacy protocol
  * built-in self-verification after writing (same criteria as the
    published-pair QC: no same-image pairs, impostors verified
    different-person against current labels, no duplicates)

Usage:
    python make_pair_files.py kdef rafd
"""

import csv
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "root_dir" / "datasets"

DATASET_LABELS = {
    "kdef": "kdef_labels.csv",
    "rafd": "rafd-frontal_aligned_labels.csv",
    "mug-still": "mug-still_labels.csv",
    "celeba-test": "celeba-test_labels.csv",
    # legacy name also present:
    "rafd-frontal_aligned": "rafd-frontal_aligned_labels.csv",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_existing_stems(ds):
    d = ROOT / "aligned" / ds
    return {p.stem for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS}


def load_identity_map(ds):
    fn = DATASET_LABELS[ds]
    m = {}
    with open(ROOT / "labels" / fn, newline="", encoding="utf-8",
              errors="replace") as f:
        for r in csv.DictReader(f):
            stem = (r.get("Name") or "").rsplit(".", 1)[0]
            m[stem] = (r.get("Identity") or "").strip()
    return m


def generate(ds, seed=42):
    rng = random.Random(seed)
    existing = load_existing_stems(ds)
    idmap = load_identity_map(ds)

    per_id = defaultdict(set)
    for stem in existing:
        ident = idmap.get(stem, "")
        if ident:
            per_id[ident].add(stem)
    unlabeled = len(existing) - sum(len(v) for v in per_id.values())
    print(f"{ds}: {len(existing)} existing images, {len(per_id)} identities, "
          f"{unlabeled} unlabeled (excluded)")

    # ---- genuine: all combinations of distinct images per identity ----
    genuine = []
    for ident in sorted(per_id):
        imgs = sorted(per_id[ident])
        genuine.extend((ident, a, b) for a, b in combinations(imgs, 2))
    print(f"  genuine pairs available: {len(genuine)}")
    rng.shuffle(genuine)

    # ---- impostor: between different identities, no duplicate rows ----
    ident_list = sorted(per_id)
    imgs_by_ident = {k: sorted(v) for k, v in per_id.items()}
    impostor, seen, tries = [], set(), 0
    while len(impostor) < len(genuine) and tries < 50 * len(genuine):
        tries += 1
        ia, ib = rng.sample(ident_list, 2)
        a = rng.choice(imgs_by_ident[ia])
        b = rng.choice(imgs_by_ident[ib])
        key = frozenset((a, b))
        if key in seen:
            continue
        seen.add(key)
        impostor.append((ia, a, ib, b))
    print(f"  impostor pairs sampled: {len(impostor)} (tries={tries})")

    pairs_dir = ROOT / "pairs"
    pairs_dir.mkdir(exist_ok=True)

    def write(name, rows):
        p = pairs_dir / name
        with open(p, "w") as f:
            for row in rows:
                f.write(f"{row[0]} {row[1]}.jpg {row[2]} {row[3] if len(row) > 3 else row[1]}\n")
        return p

    # legacy format: "<id1> <image1.jpg> <id2> <image2.jpg>"
    gfile = pairs_dir / f"{ds}_genuine_pairs.txt"
    with open(gfile, "w") as f:
        for ident, a, b in genuine:
            f.write(f"{ident} {a}.jpg {ident} {b}.jpg\n")
    ifile = pairs_dir / f"{ds}_impostor_pairs.txt"
    with open(ifile, "w") as f:
        for ia, a, ib, b in impostor:
            f.write(f"{ia} {a}.jpg {ib} {b}.jpg\n")
    print(f"  wrote {gfile.name} ({len(genuine)}) and {ifile.name} ({len(impostor)})")
    return gfile, ifile


def verify(ds, gfile, ifile):
    existing = load_existing_stems(ds)
    idmap = load_identity_map(ds)

    def rows(p):
        return [l.split() for l in open(p) if l.strip()]

    print(f"  verify {ds}:")
    for kind, p, expect_same in (("genuine", gfile, True), ("impostor", ifile, False)):
        rs = rows(p)
        same_img = sum(1 for q in rs if q[1].rsplit('.', 1)[0] == q[3].rsplit('.', 1)[0])
        missing = sum(1 for q in rs
                      if q[1].rsplit('.', 1)[0] not in existing
                      or q[3].rsplit('.', 1)[0] not in existing)
        same_id = sum(1 for q in rs
                      if idmap.get(q[1].rsplit('.', 1)[0])
                      == idmap.get(q[3].rsplit('.', 1)[0]))
        stale = sum(1 for q in rs
                    if q[0] != idmap.get(q[1].rsplit('.', 1)[0], q[0])
                    or q[2] != idmap.get(q[3].rsplit('.', 1)[0], q[2]))
        seen, dups = set(), 0
        for q in rs:
            k = frozenset((q[1], q[3]))
            dups += k in seen
            seen.add(k)
        want = (same_img == 0 and missing == 0 and stale == 0 and dups == 0
                and (same_id == len(rs) if expect_same else same_id == 0))
        print(f"    {kind:<9} n={len(rs)} same-image={same_img} missing={missing} "
              f"same-identity={'yes' if (same_id == len(rs)) else same_id} "
              f"stale-id={stale} dups={dups} -> {'OK' if want else 'FAIL'}")
        if not want:
            sys.exit(1)


def main():
    for ds in sys.argv[1:]:
        print(f"== {ds}")
        verify_existing = (ROOT / "pairs" / f"{ds}_genuine_pairs.txt").exists()
        if verify_existing and "--force" not in sys.argv:
            print("  pair file already exists - skipping (use --force to overwrite)")
            continue
        g, i = generate(ds)
        verify(ds, g, i)


if __name__ == "__main__":
    main()
