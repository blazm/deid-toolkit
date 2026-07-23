# Preprocessing Models

## shape_predictor_68_face_landmarks.dat

68-point facial landmark detector model used by dlib for precise face alignment.

**Source:** [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

Included in this repo so `deid preprocess` works out-of-the-box without manual model downloads.

The alignment pipeline falls back to MTCNN-only detection (bounding box only, no landmarks) if this file is missing.
