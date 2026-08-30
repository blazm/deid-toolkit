#!/usr/bin/env python3
"""Gender preservation for CPP-DeID fri portraits."""
import os, sys
import torch
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SCRIPt_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPt_DIR)
from extract_attributes_swinface import preprocess_image, build_swinface_model

ALIGNED = Path(r"D:\dev\deid-toolkit\root_dir\datasets\aligned\fri")
CPP_DEID = Path(r"D:\dev\deid-toolkit\root_dir\datasets\CPP-DeID\fri")

def find_image(directory, stem):
    for ext in [".png", ".PNG", ".jpg", ".jpeg", ".bmp"]:
        p = directory / (stem + ext)
        if p.exists():
            return p
    return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_swinface_model(
        os.path.join(os.path.dirname(__file__), "models", "swinface", "checkpoint_step_79999_gpu_0.pt"),
        device
    )
    model.eval()

    stems = sorted([p.stem for p in ALIGNED.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    print(f"Scanning {len(stems)} fri subjects...")

    pairs = [(find_image(ALIGNED, s), find_image(CPP_DEID, s)) for s in stems]
    valid = [(a, d) for a, d in pairs if a and d]
    missing = [stems[i] for i in range(len(pairs)) if not pairs[i][0] or not pairs[i][1]]
    print(f"Matching: {len(valid)}, Missing CPP-DeID: {missing}")

    preds_a, preds_d = [], []
    with torch.no_grad():
        # Aligned originals
        for start in range(0, len(valid), 32):
            batch = [preprocess_image(p[0]) for p in valid[start:start+32]]
            t = torch.stack(batch).to(device)
            out = model(t)
            preds_a.extend(out["Gender"].argmax(dim=1).cpu().numpy())
        # De-identified
        for start in range(0, len(valid), 32):
            batch = [preprocess_image(p[1]) for p in valid[start:start+32]]
            t = torch.stack(batch).to(device)
            out = model(t)
            preds_d.extend(out["Gender"].argmax(dim=1).cpu().numpy())

    labels_a = ["Male" if g == 1 else "Female" for g in preds_a]
    labels_d = ["Male" if g == 1 else "Female" for g in preds_d]
    matches = sum(1 for a, d in zip(labels_a, labels_d) if a == d)
    rate = matches / len(valid) * 100

    print(f"\nCPP-DeID Fri Gender Analysis:")
    print(f"  Match rate: {rate:.1f}% ({matches}/{len(valid)})")
    print(f"  {'Subject':25s} {'Aligned':6s} -> {'DeID':6s}  {'Status'}")
    for i in range(len(valid)):
        status = "SAME" if labels_a[i] == labels_d[i] else f"{labels_a[i]}->{labels_d[i]}"
        print(f"  {stems[i]:25s} {labels_a[i]:6s} -> {labels_d[i]:6s}  {status}")

if __name__ == "__main__":
    main()
