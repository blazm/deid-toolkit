#!/usr/bin/env python3
"""
UniFace Gender & Emotion Evaluation on DEID Baselines.

Runs UniFace (yakhyo/uniface v4.0.0, MIT) age-gender + emotion prediction on
aligned (original) faces and their anonymized versions across multiple DEID
techniques, then compares against ground-truth labels.

Models:
  - AgeGender: ONNX model (repo default weights), 2-way gender output
     (postprocess: gender 0=Female, 1=Male) + age. Uses face.bbox for its own
     center-alignment crop (1.5x bbox scale).
  - Emotion: TorchScript AFFECNET7 model (112x112), 7 classes
     [Neutral,Happy,Sad,Surprise,Fear,Disgust,Angry] -> mapped onto the
     SwinFace label order [Angry,Disgust,Fear,Happy,Sad,Surprise,Neutral] so
     indices are comparable across models.

Preprocessing — Protocol A (detector-free, consistent with SwinFace /
FaceXFormer / DeepFace):
  UniFace's released FaceAnalyzer API normally runs SCRFD first, but the
  attribute models only need a `Face` object (bbox + 5-point landmarks).  We
  construct one synthetically from the image frame:
    - bbox = full frame (aligned images are face-centered by the dataset
      pipeline)
    - landmarks = canonical frontal 5-point template (proportional to frame
      size)
  Both are deterministic functions of image size ONLY — no detector output is
  involved, so original and de-identified images get exactly the same
  (UniFace's own) crops: AgeGender center-aligns on the bbox (1.5x scale);
  Emotion similarity-aligns the landmark template to the 112x112 AFFECNET7
  template.  No detection misses possible.

  A `--use-detector` flag keeps the canonical SCRFD pipeline (Protocol B)
  available for reference runs where detectability is the quantity of interest.

Weight storage: all UniFace downloads are redirected to the repo via
  uniface.model_store.set_cache_dir(models/uniface) — nothing in user home.

Outputs (same convention as the other models):
  - `gender_results_uniface.csv`, `expression_results_uniface.csv`
  - `predictions_per_image_uniface.csv` (Technique, Stem, Gender, ExpressionIdx)

Usage:
    C:\\Users\\b\\anaconda3\\envs\\uniface\\python.exe evaluate_gender_expression_uniface.py ^
        --dataset kdef --output D:\\dev\\deid-toolkit\\root_dir\\predictions\\kdef
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_BASE = Path(SCRIPT_DIR).parent / "root_dir" / "datasets"
UNI_CACHE = os.path.join(SCRIPT_DIR, "models", "uniface")

import uniface  # noqa: E402  (env: uniface, PyPI 4.0.0 == cloned source)
uniface.model_store.set_cache_dir(UNI_CACHE)

from uniface import AgeGender, Emotion  # noqa: E402
from uniface.types import Face  # noqa: E402

# expression index space shared with SwinFace
EXPRESSION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
UNIFACE_EMOTION_TO_IDX = {
    "Angry": 0, "Disgust": 1, "Fear": 2, "Happy": 3, "Sad": 4, "Surprise": 5, "Neutral": 6,
}

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
                expr_idx = EXPRESSION_LABELS.index(label)
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
    reserved = ("aligned", "labels", "original", "pairs", "deidentified")
    results = []
    for tech in (technique_names or sorted(
            set(p.name for p in LABELS_BASE.iterdir()
                if p.is_dir() and p.name not in reserved))):
        tech_dir = LABELS_BASE / tech / dataset_name
        if tech_dir.is_dir():
            images = list(tech_dir.glob("*.jpg")) + list(tech_dir.glob("*.png"))
            if images:
                results.append((tech, str(tech_dir)))
    results.sort(key=lambda x: x[0])
    return results


# ── Inference ──────────────────────────────────────────────────────────────

def build_predictors():
    """AgeGender + Emotion predictors (no detector, no recognizer)."""
    return AgeGender(), Emotion()


def build_stub_face(image):
    """Protocol A: synthetic full-frame Face with a canonical frontal 5-point
    landmark template.  Depends ONLY on image size — identical preprocessing
    for originals and de-identified images, no detector in the loop."""
    h, w = image.shape[:2]
    bbox = np.array([0, 0, w, h], dtype=np.float32)
    # canonical frontal proportions: left eye, right eye, nose, left mouth, right mouth
    landmarks = np.array([
        [0.36 * w, 0.40 * h],
        [0.64 * w, 0.40 * h],
        [0.50 * w, 0.55 * h],
        [0.415 * w, 0.72 * h],
        [0.585 * w, 0.72 * h],
    ], dtype=np.float32)
    return Face(bbox=bbox, confidence=1.0, landmarks=landmarks)


def analyze_one(image, age_gender, emotion, use_detector, analyzer):
    """Returns (gender, expr_idx) for one BGR image, or (None, None) if
    detection-based mode found no face."""
    if use_detector:
        faces = analyzer.analyze(image)
        if not faces:
            return None, None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    else:
        face = build_stub_face(image)
        age_gender.predict(image, face)
        emotion.predict(image, face)
    return ("Male" if int(face.gender) == 1 else "Female",
            UNIFACE_EMOTION_TO_IDX.get(getattr(face, "emotion", None)))


def predict_batch(image_stems, image_dir, age_gender, emotion, use_detector,
                  analyzer=None, batch_size=32):
    """UniFace inference (per-image, no native batching).

    Returns dict: {stem: {"gender": "Male"/"Female", "expression_idx": int|None}}
    In detector-free mode every loadable image yields a prediction; in
    detector mode stems with no detected face are absent (misses reported).
    """
    results = {}
    paths = []
    for stem in image_stems:
        img_path = find_image_path(image_dir, stem)
        if img_path is not None:
            paths.append((stem, img_path))

    misses = 0
    t0 = time.time()
    for i, (stem, img_path) in enumerate(paths):
        img = None
        try:
            img = cv2.imread(str(img_path))
        except (ValueError, OSError):
            pass
        if img is None:
            misses += 1
            continue
        try:
            gender, expr_idx = analyze_one(img, age_gender, emotion, use_detector, analyzer)
        except Exception as e:  # noqa: BLE001 — keep the batch alive
            print(f"    ERROR on {stem}: {e}", flush=True)
            gender, expr_idx = None, None
        if gender is None:
            misses += 1
            continue
        results[stem] = {"gender": gender, "expression_idx": expr_idx}

        if (i + 1) % batch_size == 0 or (i + 1) == len(paths):
            label = f", {misses} detection misses" if use_detector else ""
            print(f"    {i + 1}/{len(paths)} images "
                  f"({(i + 1) / (time.time() - t0 + 1e-6):.1f} img/s{label})", flush=True)

    return results


# ── Evaluation & reporting (same metrics as the other models) ─────────────

def evaluate_technique(tech_name, aligned_results, deid_results, gt,
                       n_aligned_attempted=None, n_deid_attempted=None):
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

    ec = [s for s in common if gt[s]["expression"] is not None
          and deid_results[s]["expression_idx"] is not None]
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
    # UniFace reports its own detection first (Protocol B): record detectability
    res["aligned_detection_rate"] = (len(aligned_results) / n_aligned_attempted * 100
                                     if n_aligned_attempted else None)
    res["deid_detection_rate"] = (len(deid_results) / n_deid_attempted * 100
                                  if n_deid_attempted else None)
    return res


def _pct(v):
    return f"{v:.1f}" if v is not None else "N/A"


def _drate(r, key):
    v = r.get(key)
    return f"{v:.1f}" if v is not None else "N/A"


def write_csvs(results, output_dir):
    with open(str(output_dir / "gender_results_uniface.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Technique", "Aligned_Gender_Accuracy", "DeID_Gender_Accuracy",
                    "Gender_Preservation_Rate", "N_Matched_GT",
                    "Aligned_Detection_Rate", "DeID_Detection_Rate"])
        b = results[0]
        w.writerow(["aligned (Validation)", _pct(b["aligned_gender_acc"]), "—", "—", b["n_matched"],
                    _drate(b, "aligned_detection_rate"), "—"])
        for r in results[1:]:
            w.writerow([r["name"], _pct(r["aligned_gender_acc"]), _pct(r["deid_gender_acc"]),
                        _pct(r["gender_preservation_rate"]), r["n_matched"],
                        _drate(r, "aligned_detection_rate"), _drate(r, "deid_detection_rate")])

    with open(str(output_dir / "expression_results_uniface.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Technique", "Aligned_Expr_Accuracy", "DeID_Expr_Accuracy",
                    "Expr_Preservation_Rate", "N_Matched_GT",
                    "Aligned_Detection_Rate", "DeID_Detection_Rate"])
        b = results[0]
        w.writerow(["aligned (Validation)", _pct(b["aligned_expr_acc"]), "—", "—", b["n_matched"],
                    _drate(b, "aligned_detection_rate"), "—"])
        for r in results[1:]:
            w.writerow([r["name"], _pct(r["aligned_expr_acc"]), _pct(r["deid_expr_acc"]),
                        _pct(r["expr_preservation_rate"]), r["n_matched"],
                        _drate(r, "aligned_detection_rate"), _drate(r, "deid_detection_rate")])


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UniFace gender & emotion evaluation")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--techniques", nargs="+", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="progress-report interval (no native batching)")
    parser.add_argument("--use-detector", action="store_true",
                        help="Protocol B: canonical UniFace SCRFD pipeline "
                             "(records detection misses); default is detector-free Protocol A")
    args = parser.parse_args()

    dataset_name = args.dataset
    proto = ("B (canonical — SCRFD detection)" if args.use_detector
             else "A (detector-free — synthetic full-frame face + canonical landmark template)")
    print("=" * 70)
    print(f"UniFace Gender & Emotion Evaluation — {dataset_name.upper()}")
    print("=" * 70)
    print(f"Protocol: {proto}")
    print(f"Model cache: {UNI_CACHE}")

    loader_func, labels_filename = DATASET_LOADERS[dataset_name]
    gt = loader_func(str(LABELS_BASE / "labels" / labels_filename))
    n_gender = sum(1 for v in gt.values() if v["gender"])
    print(f"GT loaded: {len(gt)} images ({n_gender} with gender)")
    if n_gender == 0:
        sys.exit("ERROR: no gender GT")

    print("\nLoading UniFace models (downloads to repo cache on first run)...")
    t0 = time.time()
    age_gender, emotion = build_predictors()
    analyzer = None
    if args.use_detector:
        from uniface import FaceAnalyzer  # lazy: only Protocol B pays SCRFD's cost
        analyzer = FaceAnalyzer(predictors=[], recognizer=None)
    print(f"Predictors ready in {time.time() - t0:.1f}s")

    aligned_dir = str(LABELS_BASE / "aligned" / dataset_name)
    if not os.path.isdir(aligned_dir):
        sys.exit(f"ERROR: {aligned_dir} not found")
    eval_stems = [s for s in sorted(gt.keys()) if s in set(get_image_stems(aligned_dir))]
    print(f"\nEvaluating on {len(eval_stems)} image(s).")

    print("\nPredicting on aligned originals...")
    aligned_results = predict_batch(eval_stems, aligned_dir, age_gender, emotion,
                                    args.use_detector, analyzer, args.batch_size)
    print(f"  Aligned: {len(aligned_results)}/{len(eval_stems)}")

    tech_dirs = discover_technique_dirs(args.techniques, dataset_name)
    if not tech_dirs:
        sys.exit("ERROR: no technique dirs")
    print(f"\nFound {len(tech_dirs)} technique(s): {', '.join(t[0] for t in tech_dirs)}\n")

    results = [evaluate_technique("aligned", aligned_results, aligned_results, gt,
                                  n_aligned_attempted=len(eval_stems))]
    all_preds = {"aligned": aligned_results}

    for tech_name, tech_dir in tech_dirs:
        print(f"--- {tech_name} ---", flush=True)
        deid_stems = sorted(set(get_image_stems(tech_dir)) & set(eval_stems))
        if not deid_stems:
            print("  SKIP — no matching images.")
            continue
        deid_results = predict_batch(deid_stems, tech_dir, age_gender, emotion,
                                     args.use_detector, analyzer, args.batch_size)
        print(f"  De-identified: {len(deid_results)}/{len(deid_stems)}")
        all_preds[tech_name] = deid_results
        r = evaluate_technique(tech_name, aligned_results, deid_results, gt,
                               n_aligned_attempted=len(eval_stems),
                               n_deid_attempted=len(deid_stems))
        results.append(r)
        print(f"  Gender Acc={_pct(r['deid_gender_acc'])}%, Expr Acc={_pct(r['deid_expr_acc'])}%, "
              f"Pres G={_pct(r['gender_preservation_rate'])}% / E={_pct(r['expr_preservation_rate'])}%, "
              f"N={r['n_matched']}", flush=True)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(str(output_dir / "predictions_per_image_uniface.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Technique", "Stem", "Gender", "ExpressionIdx"])
        for tech_key in sorted(all_preds):
            for stem in sorted(all_preds[tech_key]):
                rec = all_preds[tech_key][stem]
                w.writerow([tech_key, stem, rec["gender"],
                            rec["expression_idx"] if rec["expression_idx"] is not None else "NA"])

    write_csvs(results, output_dir)
    print(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
