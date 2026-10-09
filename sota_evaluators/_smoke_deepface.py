import sys, time
import numpy as np
import cv2

print("TF GPU test:", end=" ", flush=True)
import tensorflow as tf
gpu_available = len(tf.config.list_physical_devices("GPU")) > 0
print(gpu_available, "| TF", tf.__version__)

from deepface import DeepFace

img = cv2.imread(r"D:\dev\deid-toolkit\root_dir\datasets\aligned\kdef\ANG_1_16.jpg")  # known Male
assert img is not None, "image not found"

t0 = time.time()
res = DeepFace.analyze(
    img,
    actions=("gender", "emotion"),
    detector_backend="skip",
    enforce_detection=False,
    align=False,
    silent=True,
)
print(f"first call (incl. weight download) {time.time()-t0:.1f}s")
r = res[0]
print("gender:", r.get("dominant_gender"), r.get("gender"))
print("emotion:", r.get("dominant_emotion"), {k: round(v, 3) for k, v in r.get("emotion", {}).items()})

# timing steady-state
t0 = time.time()
for _ in range(10):
    DeepFace.analyze(img, actions=("gender", "emotion"), detector_backend="skip",
                     enforce_detection=False, align=False, silent=True)
print(f"steady-state: {10/(time.time()-t0):.1f} img/s (both models, one image)")
