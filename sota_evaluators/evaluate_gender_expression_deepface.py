#!/usr/bin/env python3
"""
DeepFace Gender & Expression Evaluation on DEID Baselines.

Runs DeepFace (serengil/deepface, v0.0.101) gender + emotion prediction on
aligned (original) faces and their anonymized versions across multiple DEID
techniques, then compares against ground-truth labels.

Models (Keras/TF, CPU on this host — no Windows CUDA wheel for TF 2.21):
  - Gender:  VGGFace + softmax head, weights `gender_model_weights.h5`
             (serengil/deepface_models v1.0). Labels: ["Woman","Man"].
  - Emotion: VGG-Face-EMOTIONS, weights `facial_expression_model_weights.h5`.
             Labels: ["angry","disgust","fear","happy","sad","surprise","neutral"]
             — SAME 7 classes AND SAME ORDER as SwinFace, so expression indices
             are directly comparable across models.

Preprocessing — Protocol A (matches SwinFace / FaceXFormer-Option-A):
  detector_backend="skip" + align=False, i.e. the aligned image (already
  centered/cropped by the dataset pipeline) is fed straight through:
  cv2.imread (BGR) -> letterbox resize_image(224,224) (black padding,
  exactly what DeepFace's own module pipeline does) -> batch predict_on_batch.

NOTE: ground-truth loaders below are COPIED VERBATIM from
evaluate_gender_expression.py (same GT filters, esp. KDEF Scream/Contempt
exclusion) so N_Matched sets are identical to the SwinFace run. Keep in sync.

Outputs (same convention as the other models):
  - `gender_results_deepface.csv`, `expression_results_deepface.csv`
  - `predictions_per_image_deepface.csv` (Technique, Stem, Gender, ExpressionIdx)

Usage:
    set PYTHONUTF8=1
    C:\\Users\\b\\anaconda3\\envs\\deepface\\python.exe evaluate_gender_expression_deepface.py ^
        --dataset kdef --output D:\\dev\\deid-toolkit\\root_dir\\predictions\\kdef
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_BASE = Path(SCRIPT_DIR).parent / "root_dir" / "datasets"


# ── Expression labels (identical set & order as SwinFace) ─────────────────

EXPRESSION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EXPR_IDX_MAP = {label: i for i, label in enumerate(EXPRESSION_LABELS)}

MUG_EXPR_CODE_TO_IDX = {0: 6, 1: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5}
KDEF_EXPR_CODE_TO_IDX = {0: 6, 1: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5}


# ── GT loaders (copied verbatim from evaluate_gender_expression.py) ──────

def _read_csv(path):
    if not os.path.exists(path):
        print(f"  WARNING: Labels file not found: {path}")
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def load_rafd_labels(labels_path):
    gt = {}
    for row in _read_csv(labels_path):
        name = row.get("Name", "").strip()
        if not name:
            continue
        stem = os.path.splitext(name)[0]
        gender_code = row.get("Gender_code", "").strip()
        if gender_code == "1":
            gender = "Male"
        elif gender_code == "-1":
            gender = "Female"
        else:
            continue
        expr_idx = None
        for label in EXPRESSION_LABELS:
            col_name = label if label != "Sad" else "Sadness"
            if row.get(col_name, "").strip() == "1":
                expr_idx = EXPR_IDX_MAP[label]
                break
        gt[stem] = {"gender": gender, "expression": expr_idx}
    return gt


def load_mug_labels(labels_path):
    gt = {}
    for row in _read_csv(labels_path):
        name = row.get("Name", "").strip()
        if not name:
            continue
        stem = os.path.splitext(name)[0]
        g = row.get("Gender", "").strip()
        gender = {"M": "Male", "F": "Female"}.get(g)
        expr_idx = None
        emo_code_str = row.get("Emotion_code", "").strip()
        if emo_code_str:
            try:
                expr_idx = MUG_EXPR_CODE_TO_IDX.get(int(emo_code_str))
            except ValueError:
                pass
        gt[stem] = {"gender": gender, "expression": expr_idx}
    return gt


def load_kdef_labels(labels_path):
    gt = {}
    for row in _read_csv(labels_path):
        name = row.get("Name", "").strip()
        if not name:
            continue
        stem = os.path.splitext(name)[0]
        g = row.get("Gender", "").strip()
        gender = {"M": "Male", "F": "Female"}.get(g)
        if row.get("Scream", "").strip() == "1" or row.get("Contempt", "").strip() == "1":
            continue
        expr_idx = None
        emo_code_str = row.get("Emotion_code", "").strip()
        if emo_code_str:
            try:
                expr_idx = KDEF_EXPR_CODE_TO_IDX.get(int(emo_code_str))
            except ValueError:
                pass
        gt[stem] = {"gender": gender, "expression": expr_idx}
    return gt


def load_celeba_labels(labels_path):
    gt = {}
    for row in _read_csv(labels_path):
        name = row.get("Name", "").strip()
        if not name:
            continue
        stem = os.path.splitext(name)[0]
        gender_code = row.get("Gender_code", "").strip()
        if gender_code == "1":
            gender = "Male"
        elif gender_code == "-1":
            gender = "Female"
        else:
            continue
        gt[stem] = {"gender": gender, "expression": None}
    return gt


DATASET_LOADERS = {
    "rafd": (load_rafd_labels, "rafd-frontal_aligned_labels.csv"),
    "mug-still": (load_mug_labels, "mug-still_labels.csv"),
    "kdef": (load_kdef_labels, "kdef_labels.csv"),
    "celeba-test": (load_celeba_labels, "celeba-test_labels.csv"),
}


# ── Discovery helpers ──────────────────────────────────────────────────────

def find_image_path(directory, stem):
    d = Path(directory)
    for ext in (".png", ".PNG", ".jpg", ".jpeg", ".bmp"):
        candidate = d / (stem + ext)
        if candidate.exists():
            return candidate
    for p in d.iterdir():
        if p.stem == stem and p.is_file():
            return p
    return None


def get_image_stems(directory):
    stems = set()
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        for p in Path(directory).glob(f"*{ext}"):
            stems.add(p.stem)
    return sorted(stems)


def discover_technique_dirs(technique_names, dataset_name):
    datasets_base = LABELS_BASE
    reserved = ("aligned", "labels", "original", "pairs", "deidentified")
    results = []
    for tech in (technique_names or sorted(
            set(p.name for p in datasets_base.iterdir()
                if p.is_dir() and p.name not in reserved))):
        tech_dir = datasets_base / tech / dataset_name
        if tech_dir.is_dir():
            images = list(tech_dir.glob("*.jpg")) + list(tech_dir.glob("*.png"))
            if images:
                results.append((tech, str(tech_dir)))
    results.sort(key=lambda x: x[0])
    return results


# ── DeepFace model loading & batched inference ─────────────────────────────

DEEPFACE_SRC = os.path.join(SCRIPT_DIR, "DeepFace")
sys.path.insert(0, DEEPFACE_SRC)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Repo-local DeepFace weight store (models/deepface_home/.deepface/weights/*.h5)
os.environ.setdefault("DEEPFACE_HOME", os.path.join(SCRIPT_DIR, "models", "deepface_home"))

from deepface.modules import preprocessing as df_preprocessing  # noqa: E402
from deepface.models.demography.Gender import GenderClient  # noqa: E402
from deepface.models.demography.Emotion import EmotionClient  # noqa: E402

GENDER_LABELS = ["Woman", "Man"]  # index 1 = Male


def letterbox(img, size=(224, 224)):
    """Exact DeepFace module-pipeline preprocessing: letterbox with black pixels.
    (library's resize_image returns (1,H,W,C) — squeeze back to (H,W,C))"""
    out = df_preprocessing.resize_image(img=img, target_size=size)
    return out[0] if out.ndim == 4 else out


def predict_batch(image_stems, image_dir, gender_model, emotion_model, device_note,
                  batch_size=32):
    """Batched DeepFace gender + emotion inference.

    Returns dict: {stem: {"gender": "Male"/"Female", "expression_idx": int}}
    """
    results = {}
    paths = []
    for stem in image_stems:
        img_path = find_image_path(image_dir, stem)
        if img_path is not None:
            paths.append((stem, img_path))

    t0 = time.time()
    done = 0
    for start in range(0, len(paths), batch_size):
        g_batch, batch_stems = [], []
        for stem, img_path in paths[start:start + batch_size]:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                g_batch.append(letterbox(img))
                batch_stems.append(stem)
            except (ValueError, OSError):
                pass
        if not g_batch:
            continue

        batch = np.stack(g_batch)  # (n,224,224,3)
        gender_probs = gender_model.predict(batch)   # (n,2): [Woman, Man]
        emotion_probs = emotion_model.predict(batch)  # (n,7): SwinFace-compatible order

        for j, stem in enumerate(batch_stems):
            gender = "Male" if int(np.argmax(gender_probs[j])) == 1 else "Female"
            results[stem] = {
                "gender": gender,
                "expression_idx": int(np.argmax(emotion_probs[j])),
            }
        done += len(batch_stems)
        print(f"    {done}/{len(paths)} images "
              f"({done / (time.time() - t0 + 1e-6):.1f} img/s)", flush=True)

    return results


# ── Evaluation & reporting (same metrics as the other models) ─────────────

def evaluate_technique(tech_name, aligned_results, deid_results, gt):
    common = [s for s in sorted(set(aligned_results) & set(deid_results) & set(gt))
              if gt[s]["gender"] is not None]
    n = len(common)
    res = {"name": tech_name, "n_matched": n}
    if n:
        res["aligned_gender_acc"] = (
            sum(1 for s in common if aligned_results[s]["gender"] == gt[s]["gender"]) / n * 100)
        res["deid_gender_acc"] = (
            sum(1 for s in common if deid_results[s]["gender"] == gt[s]["gender"]) / n * 100)
        res["gender_preservation_rate"] = (
            sum(1 for s in common if aligned_results[s]["gender"] == deid_results[s]["gender"]) / n * 100)
    else:
        res["aligned_gender_acc"] = res["deid_gender_acc"] = res["gender_preservation_rate"] = None

    # expression: only stems with expression GT
    ec = [s for s in common if gt[s]["expression"] is not None]
    ne = len(ec)
    if ne:
        res["aligned_expr_acc"] = (
            sum(1 for s in ec if aligned_results[s]["expression_idx"] == gt[s]["expression"]) / ne * 100)
        res["deid_expr_acc"] = (
            sum(1 for s in ec if deid_results[s]["expression_idx"] == gt[s]["expression"]) / ne * 100)
        res["expr_preservation_rate"] = (
            sum(1 for s in ec if aligned_results[s]["expression_idx"] == deid_results[s]["expression_idx"]) / ne * 100)
    else:
        res["aligned_expr_acc"] = res["deid_expr_acc"] = res["expr_preservation_rate"] = None
    return res


def _pct(v):
    return f"{v:.1f}" if v is not None else "N/A"


def write_csvs(results, output_dir):
    with open(str(output_dir / "gender_results_deepface.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Technique", "Aligned_Gender_Accuracy", "DeID_Gender_Accuracy",
                    "Gender_Preservation_Rate", "N_Matched_GT"])
        b = results[0]
        w.writerow(["aligned (Validation)", _pct(b["aligned_gender_acc"]), "—", "—", b["n_matched"]])
        for r in results[1:]:
            w.writerow([r["name"], _pct(r["aligned_gender_acc"]), _pct(r["deid_gender_acc"]),
                        _pct(r["gender_preservation_rate"]), r["n_matched"]])

    with open(str(output_dir / "expression_results_deepface.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Technique", "Aligned_Expr_Accuracy", "DeID_Expr_Accuracy",
                    "Expr_Preservation_Rate", "N_Matched_GT"])
        b = results[0]
        w.writerow(["aligned (Validation)", _pct(b["aligned_expr_acc"]), "—", "—", b["n_matched"]])
        for r in results[1:]:
            w.writerow([r["name"], _pct(r["aligned_expr_acc"]), _pct(r["deid_expr_acc"]),
                        _pct(r["expr_preservation_rate"]), r["n_matched"]])


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DeepFace gender & expression evaluation")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--techniques", nargs="+", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    dataset_name = args.dataset
    print("=" * 70)
    print(f"DeepFace Gender & Expression Evaluation — {dataset_name.upper()}")
    print("=" * 70)
    print(f"Preprocessing: Protocol A (detector skip, no alignment, letterbox 224)")

    loader_func, labels_filename = DATASET_LOADERS[dataset_name]
    gt = loader_func(str(LABELS_BASE / "labels" / labels_filename))
    n_gender = sum(1 for v in gt.values() if v["gender"])
    print(f"GT loaded: {len(gt)} images ({n_gender} with gender)")
    if n_gender == 0:
        sys.exit("ERROR: no gender GT")

    print("\nLoading DeepFace Gender + Emotion models (CPU TF)...")
    t0 = time.time()
    gender_model = GenderClient()
    emotion_model = EmotionClient()
    print(f"Models ready in {time.time() - t0:.1f}s")

    aligned_dir = str(LABELS_BASE / "aligned" / dataset_name)
    if not os.path.isdir(aligned_dir):
        sys.exit(f"ERROR: {aligned_dir} not found")
    eval_stems = [s for s in sorted(gt.keys()) if s in set(get_image_stems(aligned_dir))]
    print(f"\nEvaluating on {len(eval_stems)} image(s).")

    print("\nPredicting on aligned originals...")
    aligned_results = predict_batch(eval_stems, aligned_dir, gender_model, emotion_model,
                                    "cpu", args.batch_size)
    print(f"  Aligned: {len(aligned_results)}/{len(eval_stems)}")

    tech_dirs = discover_technique_dirs(args.techniques, dataset_name)
    if not tech_dirs:
        sys.exit("ERROR: no technique dirs")
    print(f"\nFound {len(tech_dirs)} technique(s): {', '.join(t[0] for t in tech_dirs)}\n")

    results = [evaluate_technique("aligned", aligned_results, aligned_results, gt)]
    all_preds = {"aligned": aligned_results}

    for tech_name, tech_dir in tech_dirs:
        print(f"--- {tech_name} ---", flush=True)
        deid_stems = sorted(set(get_image_stems(tech_dir)) & set(eval_stems))
        if not deid_stems:
            print("  SKIP — no matching images.")
            continue
        deid_results = predict_batch(deid_stems, tech_dir, gender_model, emotion_model,
                                     "cpu", args.batch_size)
        print(f"  De-identified: {len(deid_results)}/{len(deid_stems)}")
        all_preds[tech_name] = deid_results
        r = evaluate_technique(tech_name, aligned_results, deid_results, gt)
        results.append(r)
        print(f"  Gender Acc={_pct(r['deid_gender_acc'])}%, Expr Acc={_pct(r['deid_expr_acc'])}%, "
              f"Pres G={_pct(r['gender_preservation_rate'])}% / E={_pct(r['expr_preservation_rate'])}%, "
              f"N={r['n_matched']}", flush=True)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(str(output_dir / "predictions_per_image_deepface.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Technique", "Stem", "Gender", "ExpressionIdx"])
        for tech_key in sorted(all_preds):
            for stem in sorted(all_preds[tech_key]):
                rec = all_preds[tech_key][stem]
                w.writerow([tech_key, stem, rec["gender"], rec["expression_idx"]])

    write_csvs(results, output_dir)
    print(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
