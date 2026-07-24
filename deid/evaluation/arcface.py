"""ArcFace evaluation using ONNX model (no face detection — works on pre-aligned images)."""
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
from tqdm import tqdm

import utils as util

SCRIPT_DIR = Path(__file__).resolve().parent
PATH_TO_MODEL = str(SCRIPT_DIR / "weights" / "model.onnx")
EVALUATION_NAME = "arcface"


def _setup_session(model_path: str):
    """Create ONNX inference session, preferring GPU if available."""
    providers = [
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]
    return onnxruntime.InferenceSession(model_path, providers=providers)


def _preprocess(img_path: str, input_mean: float, input_std: float, input_size: tuple):
    """Load image, resize to 112x112, normalize and return blob for ONNX."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, input_size)
    blob = cv2.dnn.blobFromImage(
        img, 1.0 / input_std, input_size,
        (input_mean, input_mean, input_mean), swapRB=True,
    )
    return blob


def main():
    args = util.read_args()

    path_to_aligned_images = args.aligned_path
    path_to_deidentified_images = args.deidentified_path
    path_to_genuine_pairs = args.genuine_pairs_filepath
    path_to_impostor_pairs = args.impostor_pairs_filepath
    path_to_save = args.save_path
    path_to_log = args.dir_to_log

    dataset_name = util.get_dataset_name_from_path(path_to_aligned_images)
    technique_name = util.get_technique_name_from_path(path_to_deidentified_images)
    metrics_df = util.Metrics(name_score="cossim")

    if not path_to_impostor_pairs:
        print("No impostor pairs provided")
        return
    if not path_to_genuine_pairs:
        print("No genuine pairs provided")
        return

    # Load ONNX model
    print(f"Loading ArcFace ONNX model from {PATH_TO_MODEL}")
    session = _setup_session(PATH_TO_MODEL)
    input_cfg = session.get_inputs()[0]
    input_name = input_cfg.name
    input_size = tuple(input_cfg.shape[2:4][::-1])  # (H, W)
    output_names = [o.name for o in session.get_outputs()]

    # Detect normalization (mxnet vs arcface convention)
    import onnx
    model_proto = onnx.load(PATH_TO_MODEL)
    has_sub = any(
        n.name.startswith("Sub") or n.name.startswith("_minus")
        for n in model_proto.graph.node[:8]
        if hasattr(n, "name")
    )
    has_mul = any(
        n.name.startswith("Mul") or n.name.startswith("_mul")
        for n in model_proto.graph.node[:8]
        if hasattr(n, "name")
    )
    if has_sub and has_mul:
        input_mean, input_std = 0.0, 1.0
    else:
        input_mean, input_std = 127.5, 127.5
    print(f"  Input size: {input_size}, norm: mean={input_mean}, std={input_std}")

    # Read pairs
    genu_names_a, genu_ids_a, genu_names_b, genu_ids_b = util.read_pairs_file(path_to_genuine_pairs)
    impo_names_a, impo_ids_a, impo_names_b, impo_ids_b = util.read_pairs_file(path_to_impostor_pairs)

    names_a = genu_names_a + impo_names_a
    names_b = genu_names_b + impo_names_b
    ids_a = genu_ids_a + impo_ids_a
    ids_b = genu_ids_b + impo_ids_b
    ground_truth = np.array([int(a == b) for a, b in zip(ids_a, ids_b)])

    cos_scores = []

    for i, (name_a, name_b, gt) in enumerate(
        tqdm(zip(names_a, names_b, ground_truth), total=len(names_a),
             desc=f"arcface | {dataset_name}-{technique_name}")
    ):
        if util._TEST_SINGLE and i > 0:
            break

        path_a = os.path.join(path_to_aligned_images, name_a)
        path_b = os.path.join(path_to_deidentified_images, name_b)

        if not os.path.exists(path_a):
            util.log(os.path.join(path_to_log, "arcface.txt"),
                     f"({dataset_name}) Missing: {path_a}")
            continue
        if not os.path.exists(path_b):
            util.log(os.path.join(path_to_log, "arcface.txt"),
                     f"({technique_name}) Missing: {path_b}")
            continue

        blob_a = _preprocess(path_a, input_mean, input_std, input_size)
        blob_b = _preprocess(path_b, input_mean, input_std, input_size)

        feat_a = session.run(output_names, {input_name: blob_a})[0].flatten()
        feat_b = session.run(output_names, {input_name: blob_b})[0].flatten()

        # Cosine similarity
        norm_a = np.linalg.norm(feat_a)
        norm_b = np.linalg.norm(feat_b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            cos_sim = 0.0
        else:
            cos_sim = np.dot(feat_a, feat_b) / (norm_a * norm_b)

        cos_scores.append(cos_sim)
        metrics_df.add_score(name_a, cos_sim)
        metrics_df.add_column_value("ground_truth", gt)

    if cos_scores:
        print(f"MIN: {np.min(cos_scores):.4f}  MAX: {np.max(cos_scores):.4f}")
    else:
        print("No scores computed — check that image paths are correct.")

    metrics_df.save_to_csv(path_to_save)
    print(f"ArcFace scores saved to {path_to_save}")


if __name__ == "__main__":
    main()
