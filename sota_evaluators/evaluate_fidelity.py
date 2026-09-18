#!/usr/bin/env python
"""
Batched GPU fidelity metrics for the DeID toolkit, CelebA-HQ test set
(2,824 aligned 256x256 pairs per technique).

Metrics (all full-reference / set-to-set; NOT part of the naturalness
composite N -- companion columns for tab:naturalness):
  MSE     -- mean squared error over pixels, values in [0,1] (lower = better)
  PSNR    -- 10*log10(1/MSE) in dB for values in [0,1] (higher = better)
  SSIM    -- single-scale structural similarity, pytorch_msssim (higher = better)
  FID     -- Frechet Inception Distance, 2048-d pool3 Inception-v3 features,
             299x299 upscaled center-crop (standard), aligned vs de-identified
             image SETS (lower = better)

Pairs each technique's de-identified output with the aligned reference
original by file stem. Single source of truth:
deid-toolkit/sota_evaluators/evaluate_fidelity.py

Usage:
  python evaluate_fidelity.py                     # full: 20 techniques
  python evaluate_fidelity.py --smoke 16          # first 16 pairs of each
  python evaluate_fidelity.py --techniques AIDPro
  python evaluate_fidelity.py --metrics ssim      # ssim/mse only (skip FID)
  python evaluate_fidelity.py --resume

Output layout (one file pair per technique, inside the metric-named folder):
  <root>/root_dir/predictions/fidelity/celeba-test/<TECH>.json
    {"technique": ..., "dataset": "celeba-test", "n_pairs": ...,
     "missing": ..., "mse_mean": ..., "mse_std": ..., "psnr_db": ...,
     "ssim_mean": ..., "ssim_std": ..., "fid": ...}
  <root>/root_dir/predictions/fidelity/celeba-test/<TECH>_perimage.csv
    image,mse,psnr,ssim
No combined all-technique files; readers glob the folder.
Reference (aligned vs aligned) row: MSE 0, PSNR inf, SSIM 1, FID 0.
"""
import argparse, csv, glob, json, math, os, time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BATCH = 64
FID_BATCH = 32
FID_SIZE = 299
DATASET = "celeba-test"
METRICS = ("ssim", "fid")

GENERATIVE_TECHNIQUES = [
    "AIDPro", "AMT-GAN", "AnonNET", "CLEANIR", "CPP-DeID", "DeepPrivacy",
    "DeepPrivacy2", "DiffPrivate", "FADM", "FALCO", "FAMS", "G2Face",
    "GANonymization", "IPFA", "LDFA", "NullFace", "PRO-Face", "RP", "RiDDLE",
    "iFADIT",
]


def load01(path, size=None):
    """Path -> (C,H,W) float32 [0,1]; optional upscale to size x size with
    LANCZOS (pytorch-fid FID convention at 299)."""
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    t = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
    return t


def run_pair_metrics(paired_files):
    """(orig,deid) list -> {stem: (mse, psnr_db, ssim)} at normalized 256.
    (A few aligned crops are larger than 256, e.g. 1024: both sides are
    normalized to 256x256 before comparison.)"""
    from pytorch_msssim import SSIM
    ssim_fn = SSIM(data_range=1.0).eval().to("cuda")
    out = {}
    with torch.no_grad():
        for i in range(0, len(paired_files), BATCH):
            chunk = paired_files[i:i + BATCH]
            A = torch.stack([load01(p, size=256) for p, _ in chunk]).to("cuda")
            B = torch.stack([load01(p, size=256) for _, p in chunk]).to("cuda")
            mse = ((A - B) ** 2).mean(dim=(1, 2, 3)).cpu().numpy().tolist()
            s = ssim_fn(A, B).reshape(-1).cpu().numpy().tolist()
            for (op, _), m, sc in zip(chunk, mse, s):
                stem = os.path.splitext(os.path.basename(op))[0]
                psnr = 10.0 * math.log10(1.0 / m) if m > 0 else float("inf")
                out[stem] = (m, psnr, sc)
    return out


def inception_model(root):
    """2048-d FID features via the package's own InceptionV3 wrapper
    (block 3 = Mixed_7a/7b/7c + global avgpool; internal 299x299 resize +
    [0,1]->[-1,1] normalize). Weights are loaded from the LOCAL toolkit
    copy (monkeypatched), so the run is fully offline-safe."""
    import pytorch_fid.inception as pf
    wpath = os.path.join(root, "deid", "evaluation", "weights",
                         "inception_fid", "pt_inception-2015-12-05-6726825d.pth")

    def _local_load(url, *a, **k):
        return torch.load(wpath, map_location="cuda")

    pf.load_state_dict_from_url = _local_load
    m = pf.InceptionV3().eval().to("cuda")
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def run_features(feat_fn, paths):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), FID_BATCH):
            x = torch.stack([load01(p) for p in paths[i:i + FID_BATCH]]).to("cuda")
            f = feat_fn(x)[0].flatten(1).cpu().numpy()
            out.append(f)
    return np.concatenate(out, axis=0)


def calc_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    from scipy.linalg import sqrtm
    d = mu1 - mu2
    covmean = sqrtm(np.matmul(sigma1, sigma2))
    covmean = np.real_if_close(covmean)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.real(d @ d + np.trace(sigma1 + sigma2 - 2.0 * covmean)))


def fid_from_features(fa, fb):
    mu_a, sig_a = fa.mean(0), np.cov(fa, rowvar=False)
    mu_b, sig_b = fb.mean(0), np.cov(fb, rowvar=False)
    return calc_fid(mu_a, sig_a, mu_b, sig_b)


def tech_complete(sum_path, rows_path, metrics):
    if not (os.path.exists(sum_path) and os.path.exists(rows_path)):
        return False
    r = json.load(open(sum_path, encoding="ascii"))
    if "ssim_mean" not in r or "fid" not in r:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("DEID_ROOT", "D:/dev/deid-toolkit"))
    ap.add_argument("--techniques", nargs="*", default=None)
    ap.add_argument("--metrics", nargs="*", default=list(METRICS))
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    ds = os.path.join(args.root, "root_dir", "datasets")
    out_dir = args.out_dir or os.path.join(args.root, "root_dir",
                                           "predictions", "fidelity", DATASET)
    os.makedirs(out_dir, exist_ok=True)

    techs = args.techniques or GENERATIVE_TECHNIQUES
    orig_dir = os.path.join(ds, "aligned", DATASET)
    orig_paths = sorted(glob.glob(os.path.join(orig_dir, "*.jpg")) +
                        glob.glob(os.path.join(orig_dir, "*.png")))
    if args.smoke:
        orig_paths = orig_paths[:args.smoke]
    stems = [os.path.splitext(os.path.basename(p))[0] for p in orig_paths]
    orig_map = dict(zip(stems, orig_paths))

    feat_fn = None
    ref_feat = None
    if "fid" in args.metrics:
        feat_fn = inception_model(args.root)
        ref_path = os.path.join(out_dir, "_ref_features.npz")
        if os.path.exists(ref_path):
            ref_feat = np.load(ref_path)["features"]
            if ref_feat.shape != (len(orig_paths), 2048):
                import os as _os
                _os.remove(ref_path)
                ref_feat = None
        if ref_feat is None:
            print("extracting reference Inception features ...", flush=True)
            t0 = time.time()
            ref_feat = run_features(feat_fn, orig_paths)
            np.savez(ref_path, features=ref_feat)
            print(f"reference features {ref_feat.shape} in "
                  f"{time.time()-t0:.1f}s", flush=True)

    n_done = 0
    for k, tech in enumerate(techs, 1):
        sum_path = os.path.join(out_dir, tech + ".json")
        rows_path = os.path.join(out_dir, tech + "_perimage.csv")
        if args.resume and tech_complete(sum_path, rows_path, args.metrics):
            print(f"[{k}/{len(techs)}] {tech}: already done, skip", flush=True)
            continue
        r = json.load(open(sum_path, encoding="ascii")) \
            if os.path.exists(sum_path) else {}
        deid_dir = os.path.join(ds, tech, DATASET)
        pairs, missing = [], 0
        for s in stems:
            hits = sorted(glob.glob(os.path.join(deid_dir, s + ".*")))
            hits = [h for h in hits
                    if os.path.splitext(h)[1].lower() in (".jpg", ".png")]
            if hits:
                pairs.append((orig_map[s], hits[0]))
            else:
                missing += 1
        t0 = time.time()
        pm = run_pair_metrics(pairs)
        fid_val = None
        if "fid" in args.metrics:
            deid_paths = [p for _, p in pairs]
            fid_val = fid_from_features(ref_feat, run_features(feat_fn, deid_paths))
        mse = [v[0] for v in pm.values()]
        ssim = [v[2] for v in pm.values()]
        psnrs = [v[1] for v in pm.values()]
        payload = {"technique": tech, "dataset": DATASET,
                   "n_pairs": len(pairs), "missing": missing}
        if "ssim" in args.metrics and mse:
            payload.update({
                "mse_mean": float(np.mean(mse)), "mse_std": float(np.std(mse)),
                "psnr_db": float(np.mean([p for p in psnrs if p != float("inf")])),
                "ssim_mean": float(np.mean(ssim)),
                "ssim_std": float(np.std(ssim)),
            })
        if fid_val is not None:
            payload["fid"] = fid_val
        json.dump(payload, open(sum_path, "w", encoding="ascii"), indent=2)
        with open(rows_path, "w", newline="", encoding="ascii") as f:
            w = csv.writer(f)
            w.writerow(["image", "mse", "psnr", "ssim"])
            for stem, (m, psnr, s) in sorted(pm.items()):
                w.writerow([stem, f"{m:.8f}",
                            "inf" if math.isinf(psnr) else f"{psnr:.4f}",
                            f"{s:.6f}"])
        n_done += 1
        print(f"[{k}/{len(techs)}] {tech}: pairs={len(pairs)} missing={missing} "
              f"ssim={payload.get('ssim_mean', float('nan')):.4f} "
              f"psnr={payload.get('psnr_db', float('nan')):.2f} "
              f"fid={fid_val if fid_val is None else round(fid_val, 3)} "
              f"({time.time()-t0:.1f}s)", flush=True)
    print(f"done: {n_done} techniques in {out_dir}")


if __name__ == "__main__":
    main()
