#!/usr/bin/env python
"""
Batched GPU naturalness metrics for the DeID toolkit, CelebA-HQ test set
(2,824 aligned 256x256 pairs per technique).

Metrics:
  LPIPS (alex, 256->224 bilinear)   -- paired fidelity/drift to the original
  eDiffIQA (iResNet-18, variant S)  -- no-reference perceived quality;
                                       reported per side + retention ratio
                                       mean(deid)/mean(orig) in [0,1]

Pairs each technique's de-identified output with the aligned reference
original by file stem (extensions differ: .jpg original vs .png output).
Single source of truth: deid-toolkit/sota_evaluators/evaluate_naturalness.py

Usage:
  python evaluate_naturalness.py                      # full: 20 techniques
  python evaluate_naturalness.py --smoke 16           # first 16 pairs of each
  python evaluate_naturalness.py --techniques AIDPro DiffPrivate
  python evaluate_naturalness.py --metrics ediffiqa   # eDiffIQA only (e.g. after LPIPS pass)
  python evaluate_naturalness.py --resume             # skip techniques already complete
  DEID_ROOT=/path python evaluate_naturalness.py      # portable

Output layout (one file per technique, inside the metric-named folder):
  <root>/root_dir/predictions/naturalness/celeba-test/<TECH>.json
    {"technique": ..., "dataset": "celeba-test", "n_pairs": ...,
     "missing": ..., "lpips_mean": ..., "lpips_std": ...,
     "ediff_orig_mean": ..., "ediff_deid_mean": ...,
     "ediff_retention": ..., "ediff_delta_mean": ...}
  <root>/root_dir/predictions/naturalness/celeba-test/<TECH>_perimage.csv
    image,lpips,ediff_orig,ediff_deid
No combined all-technique files; readers glob the folder.
"""
import argparse, csv, glob, json, os, sys, time
import numpy as np
from PIL import Image

import torch
from torch.nn.functional import interpolate
import lpips

BATCH = 32
LPIPS_SIZE = 224   # VGG input
DATASET = "celeba-test"
METRICS = ("lpips", "ediffiqa")

GENERATIVE_TECHNIQUES = [
    "AIDPro", "AMT-GAN", "AnonNET", "CLEANIR", "CPP-DeID", "DeepPrivacy",
    "DeepPrivacy2", "DiffPrivate", "FADM", "FALCO", "FAMS", "G2Face",
    "GANonymization", "IPFA", "LDFA", "NullFace", "PRO-Face", "RP", "RiDDLE",
    "iFADIT",
]


def to_tensor(img: Image.Image) -> torch.Tensor:
    """HWC uint8 -> (C,H,W) float32 [-1,1], downscaled to 224 (uniform scale,
    keeps the center-aligned face intact)."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    t = torch.from_numpy(a).permute(2, 0, 1) / 127.5 - 1.0
    return interpolate(t.unsqueeze(0), size=(LPIPS_SIZE, LPIPS_SIZE),
                       mode="bilinear", align_corners=False).squeeze(0)


def run_lpips(loss_fn, paired_files):
    """paired_files: list of (orig_path, deid_path). Returns {stem: lpips}."""
    out = {}
    with torch.no_grad():
        for i in range(0, len(paired_files), BATCH):
            chunk = paired_files[i:i + BATCH]
            A = torch.stack([to_tensor(Image.open(p)) for p, _ in chunk]).to("cuda")
            B = torch.stack([to_tensor(Image.open(p)) for _, p in chunk]).to("cuda")
            d = loss_fn(A, B).reshape(-1).cpu().numpy().tolist()
            for (op, _), val in zip(chunk, d):
                out[os.path.splitext(os.path.basename(op))[0]] = float(val)
    return out


def load_ediffiqa(root):
    weights = os.path.join(root, "deid", "evaluation", "weights")
    sys.path.insert(0, weights)
    import importlib, pathlib
    mod = importlib.import_module("model.ediffiqa")
    model, transform = mod.load_ediffiqa(
        "ediffiqaS", pathlib.Path(os.path.join(weights, "ediffiqaS")), "cuda")
    return model, transform


def run_ediffiqa(model, transform, paths):
    """paths: list of image paths. Returns {stem: quality_score}."""
    out = {}
    tensors = [transform(Image.open(p).convert("RGB")) for p in paths]
    with torch.no_grad():
        for i in range(0, len(paths), BATCH):
            chunk = torch.stack(tensors[i:i + BATCH]).to("cuda")
            preds = model(chunk).cpu().numpy().flatten().tolist()
            for p, val in zip(paths[i:i + BATCH], preds):
                out[os.path.splitext(os.path.basename(p))[0]] = float(val)
    return out


def tech_complete(sum_path, rows_path, metrics):
    if not (os.path.exists(sum_path) and os.path.exists(rows_path)):
        return False
    r = json.load(open(sum_path, encoding="ascii"))
    for m in metrics:
        if m == "lpips" and "lpips_mean" not in r:
            return False
        if m == "ediffiqa" and "ediff_deid_mean" not in r:
            return False
    return True


def read_prior_rows(rows_path):
    """Read a prior per-image CSV. Returns {image: {col: float|None}}."""
    out = {}
    if not os.path.exists(rows_path):
        return out
    with open(rows_path, encoding="ascii") as f:
        for row in csv.DictReader(f):
            out[row["image"]] = {
                c: (float(row[c]) if row.get(c) not in (None, "") else None)
                for c in ("lpips", "ediff_orig", "ediff_deid")}
    return out


def save_technique(out_dir, tech, results, rows):
    """Write <TECH>.json and <TECH>_perimage.csv (one file pair per technique)."""
    os.makedirs(out_dir, exist_ok=True)
    payload = {"technique": tech, "dataset": DATASET}
    payload.update(results)
    with open(os.path.join(out_dir, tech + ".json"), "w", encoding="ascii") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, tech + "_perimage.csv"), "w",
              newline="", encoding="ascii") as f:
        w = csv.writer(f)
        w.writerow(["image", "lpips", "ediff_orig", "ediff_deid"])
        for s2 in sorted(rows):
            d2 = rows[s2]
            w.writerow([s2] + [
                (f"{d2[c]:.6f}" if d2.get(c) is not None else "")
                for c in ("lpips", "ediff_orig", "ediff_deid")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("DEID_ROOT", "D:/dev/deid-toolkit"))
    ap.add_argument("--smoke", type=int, default=0, help="only first N pairs per technique")
    ap.add_argument("--techniques", nargs="*", default=None)
    ap.add_argument("--metrics", nargs="+", default=list(METRICS), choices=list(METRICS))
    ap.add_argument("--out-dir", default=None,
                    help="override output dir (default <root>/root_dir/predictions/naturalness/<ds>)")
    ap.add_argument("--resume", action="store_true",
                    help="skip techniques already complete for --metrics")
    args = ap.parse_args()

    ds = os.path.join(args.root, "root_dir", "datasets")
    out_dir = args.out_dir or os.path.join(args.root, "root_dir", "predictions",
                                           "naturalness", DATASET)

    techs = args.techniques or GENERATIVE_TECHNIQUES
    orig_dir = os.path.join(ds, "aligned", DATASET)
    orig_paths = sorted(glob.glob(os.path.join(orig_dir, "*.jpg")) +
                        glob.glob(os.path.join(orig_dir, "*.png")))
    if args.smoke:
        orig_paths = orig_paths[:args.smoke]
    stems = [os.path.splitext(os.path.basename(p))[0] for p in orig_paths]
    orig_map = dict(zip(stems, orig_paths))

    loss_fn = ed_model = ed_transform = None
    if "lpips" in args.metrics:
        loss_fn = lpips.LPIPS(net="alex").eval().to("cuda")
    if "ediffiqa" in args.metrics:
        ed_model, ed_transform = load_ediffiqa(args.root)
    ed_orig_scores = run_ediffiqa(ed_model, ed_transform, orig_paths) \
        if "ediffiqa" in args.metrics else {}

    n_done = 0
    t0 = time.time()
    for k, tech in enumerate(techs, 1):
        sum_path = os.path.join(out_dir, tech + ".json")
        rows_path = os.path.join(out_dir, tech + "_perimage.csv")
        if args.resume and tech_complete(sum_path, rows_path, args.metrics):
            print(f"[{k}/{len(techs)}] {tech}: already done, skip", flush=True)
            continue
        r = json.load(open(sum_path, encoding="ascii")) \
            if os.path.exists(sum_path) else {}
        rows = read_prior_rows(rows_path)
        deid_dir = os.path.join(ds, tech, DATASET)
        pairs, missing = [], 0
        for s in stems:
            op = orig_map.get(s)
            dp = next((os.path.join(deid_dir, s + e) for e in (".jpg", ".png", ".jpeg")
                       if os.path.exists(os.path.join(deid_dir, s + e))), None)
            if op and dp:
                pairs.append((op, dp))
            else:
                missing += 1
        if not pairs:
            print(f"[{k}/{len(techs)}] {tech}: NO PAIRS (dir={deid_dir}), skip", flush=True)
            continue
        if "lpips" in args.metrics and "lpips_mean" not in r:
            lp = run_lpips(loss_fn, pairs)
            for s2, v2 in lp.items():
                rows.setdefault(s2, {})["lpips"] = v2
            v = np.array(list(lp.values()))
            r["lpips_mean"], r["lpips_std"] = float(v.mean()), float(v.std())
        if "ediffiqa" in args.metrics and "ediff_deid_mean" not in r:
            deid_paths = [dp for _, dp in pairs]
            ed_deid = run_ediffiqa(ed_model, ed_transform, deid_paths)
            for s2, v2 in ed_deid.items():
                rows.setdefault(s2, {})[
                    "ediff_deid"] = v2
                if s2 in ed_orig_scores:
                    rows[s2]["ediff_orig"] = ed_orig_scores[s2]
            pair_stems = [os.path.splitext(os.path.basename(op))[0] for op, _ in pairs]
            vd = np.array([ed_deid[s2] for s2 in pair_stems if s2 in ed_deid])
            vo = np.array([ed_orig_scores[s2] for s2 in pair_stems if s2 in ed_orig_scores])
            aligned = (len(vd) == len(vo) and len(vo) > 0)
            r["ediff_orig_mean"] = float(vo.mean()) if aligned else None
            r["ediff_deid_mean"] = float(vd.mean()) if len(vd) else None
            r["ediff_retention"] = (float(vd.mean() / vo.mean())
                                    if (aligned and vo.mean() > 0) else None)
            r["ediff_delta_mean"] = float((vd - vo).mean()) if aligned else None
        r["n_pairs"] = int(len(pairs))
        r["missing"] = int(missing)
        save_technique(out_dir, tech, {kk: vv for kk, vv in r.items()
                                       if kk not in ("technique", "dataset")}, rows)
        n_done += 1
        lp_s = ("lpips=%.4f" % r["lpips_mean"]) if "lpips_mean" in r else "lpips=  -  "
        rt_s = ("ret=%.3f" % r["ediff_retention"]) \
            if r.get("ediff_retention") is not None else "ret=  -  "
        print(f"[{k}/{len(techs)}] {tech:<15s} n={len(pairs):>5d} {lp_s} {rt_s} "
              f"({time.time()-t0:6.1f}s)", flush=True)

    print(f"\nDone. {n_done} techniques written into {out_dir}")
    for tech in techs:
        p = os.path.join(out_dir, tech + ".json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p, encoding="ascii"))
        lp = r.get("lpips_mean"); rt = r.get("ediff_retention")
        print(f"  {tech:<15s} "
              f"{('lpips=%.4f' % lp) if lp is not None else 'lpips=  -  '} "
              f"{('ret=%.3f' % rt) if rt is not None else 'ret=  -  '}")


if __name__ == "__main__":
    main()
