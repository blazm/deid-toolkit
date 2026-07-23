"""Multiprocessing face alignment using MTCNN.

Ported from modules/utils/align_face_mtcnn.py with current uncommitted changes:
- Multiprocessing support (mp_main)
- RAF-D filename filtering
"""
from __future__ import annotations

import numpy as np
import PIL
import PIL.Image
import os
import scipy
import scipy.ndimage
from pathlib import Path
import cv2
from mtcnn import MTCNN
from tqdm import tqdm

# Optional dlib predictor — first checks env var, then bundled package path,
# then legacy root_dir/preprocess fallback.
_predictor = None
_PREDICTOR_FALLBACKS = [
    lambda: os.environ.get("DEID_SHAPE_PREDICTOR"),
    lambda: str(Path(__file__).resolve().parent.parent / "models" / "preprocessing" / "shape_predictor_68_face_landmarks.dat"),
    lambda: "root_dir/preprocess/shape_predictor_68_face_landmarks.dat",
]
_predictor_path: str | None = None
for _fallback in _PREDICTOR_FALLBACKS:
    _predictor_path = _fallback()
    if _predictor_path and os.path.exists(_predictor_path):
        break

if _predictor_path and os.path.exists(_predictor_path):
    try:
        import dlib
        _predictor = dlib.shape_predictor(_predictor_path)
    except Exception:
        _predictor = None


def get_landmark(filepath):
    """Get landmark with dlib, fallback to MTCNN if dlib fails.
    :return: np.array shape=(68, 2)
    """
    if _predictor is not None:
        detector = dlib.get_frontal_face_detector()
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Image not found or could not be loaded: {filepath}")

    dets = detector(img, 1) if _predictor is not None else []

    if len(dets) == 0:
        return get_landmark_mtcnn(filepath)

    shape = _predictor(img, dets[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return landmarks


def get_landmark_mtcnn(filepath):
    """Get landmark with MTCNN.
    :return: np.array shape=(5, 2)
    """
    detector = MTCNN()
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Image not found or could not be loaded: {filepath}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)

    if len(results) == 0:
        raise ValueError(f"No faces detected in the image: {filepath}")

    keypoints = results[0]['keypoints']
    landmarks = np.array([
        [keypoints['left_eye'][0], keypoints['left_eye'][1]],
        [keypoints['right_eye'][0], keypoints['right_eye'][1]],
        [keypoints['nose'][0], keypoints['nose'][1]],
        [keypoints['mouth_left'][0], keypoints['mouth_left'][1]],
        [keypoints['mouth_right'][0], keypoints['mouth_right'][1]],
    ])
    return landmarks


def adapt_mtcnn_landmarks(mtcnn_landmarks):
    """Adapt MTCNN landmarks (5 points) to a format similar to dlib's 68 points."""
    left_eye = mtcnn_landmarks[0]
    right_eye = mtcnn_landmarks[1]
    nose = mtcnn_landmarks[2]
    mouth_left = mtcnn_landmarks[3]
    mouth_right = mtcnn_landmarks[4]

    landmarks = np.zeros((68, 2), dtype=np.float32)
    landmarks[36:42] = left_eye
    landmarks[42:48] = right_eye
    landmarks[30] = nose
    landmarks[48] = mouth_left
    landmarks[54] = mouth_right

    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (mouth_left + mouth_right) / 2.0
    chin_bottom = mouth_center + (mouth_center - nose) * 2
    landmarks[8] = chin_bottom

    return landmarks


def align_face(filepath):
    """Align face in the given image."""
    lm = get_landmark(filepath)

    if lm.shape[0] == 5:
        lm = adapt_mtcnn_landmarks(lm)

    lm_chin = lm[0:17]
    lm_eyebrow_left = lm[17:22]
    lm_eyebrow_right = lm[22:27]
    lm_nose = lm[27:31]
    lm_nostrils = lm[31:36]
    lm_eye_left = lm[36:42]
    lm_eye_right = lm[42:48]
    lm_mouth_outer = lm[48:60]
    lm_mouth_inner = lm[60:68]

    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = PIL.Image.open(filepath)

    output_size = 1024
    transform_size = 4096
    enable_padding = True

    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (
            int(np.rint(float(img.size[0]) / shrink)),
            int(np.rint(float(img.size[1]) / shrink)),
        )
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1]),
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    pad = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - img.size[0] + border, 0),
        max(pad[3] - img.size[1] + border, 0),
    )
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(
            np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect"
        )
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]),
        )
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(
            mask * 3.0 + 1.0, 0.0, 1.0
        )
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), "RGB")
        quad += pad[:2]

    img = img.transform(
        (transform_size, transform_size),
        PIL.Image.QUAD,
        (quad + 0.5).flatten(),
        PIL.Image.BILINEAR,
    )
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    return img


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def filter_filenames(img_names, dataset_name):
    """Filter image names based on dataset-specific patterns."""
    if dataset_name == 'rafd':
        img_names = [i for i in img_names
            if "frontal" in i
            and ("090" in i or "135" in i or "045" in i)]
    return img_names


def _chunk(file_list, num_chunks):
    """Split file list into roughly equal chunks."""
    n = len(file_list)
    return [file_list[i * n // num_chunks:(i + 1) * n // num_chunks]
            for i in range(num_chunks)]


def mp_main(img_paths=None, save_paths=None, dataset_path=None, dataset_save_path=None, dataset_name=None):
    """Multiprocessing version of main alignment."""
    dataset_filetype = "jpg"
    dataset_filetype_1 = "JPG"
    dataset_filetype_2 = "png"
    dataset_newtype = "jpg"

    if dataset_path is not None:
        img_names = [
            i for i in os.listdir(dataset_path)
            if (dataset_filetype in i or dataset_filetype_1 in i or dataset_filetype_2 in i)
        ]
        img_names = filter_filenames(img_names, dataset_name)
        img_paths = [os.path.join(dataset_path, i) for i in img_names]
        save_paths = [
            os.path.join(dataset_save_path, os.path.splitext(i)[0] + "." + dataset_newtype)
            for i in img_names]

    workers = 12
    data_chunks = _chunk(img_names, workers)
    from multiprocessing import Process
    ps = []

    for ix in range(workers):
        img_names_ix = data_chunks[ix]
        img_paths_ix = [os.path.join(dataset_path, i) for i in img_names_ix]
        save_paths_ix = [
            os.path.join(dataset_save_path, os.path.splitext(i)[0] + "." + dataset_newtype)
            for i in img_names_ix]

        w = Process(target=main, args=(img_paths_ix, save_paths_ix, dataset_path, dataset_save_path, dataset_name))
        w.daemon = True
        w.start()
        ps.append(w)

    for w in ps:
        w.join()


def main(img_paths=None, save_paths=None, dataset_path=None, dataset_save_path=None, dataset_name=None):
    dataset_filetype = "jpg"
    dataset_filetype_1 = "JPG"
    dataset_filetype_2 = "png"
    dataset_newtype = "jpg"

    if dataset_path is not None and img_paths is None and save_paths is None:
        img_names = [
            i for i in os.listdir(dataset_path)
            if (dataset_filetype in i or dataset_filetype_1 in i or dataset_filetype_2 in i)
        ]
        img_names = filter_filenames(img_names, dataset_name)
        img_paths = [os.path.join(dataset_path, i) for i in img_names]
        save_paths = [
            os.path.join(dataset_save_path, os.path.splitext(i)[0] + "." + dataset_newtype)
            for i in img_names]

    count = 0
    for img_path, save_path in tqdm(zip(img_paths, save_paths), total=len(img_paths),
                                     desc=f"Processing images of {dataset_name}", unit="it"):
        try:
            img = align_face(img_path)
            ensure_dir(save_path)
            img.save(save_path, "JPEG")
            count += 1
        except Exception as e:
            print(f"Failed to process image {img_path}: {e}")
            print("Img path is: " + img_path)
