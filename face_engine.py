"""
face_engine.py - finds faces and turns each one into a 128-number "face print".

Uses two small models that ship with OpenCV's model zoo (no extra pip packages):
  * YuNet  - detects faces
  * SFace  - converts a face into 128 numbers; the same person gives similar numbers

The models (~38 MB total) are downloaded once into the  models/  folder.
"""

import os
import threading
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

MODEL_DIR = "models"
BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx", 100_000),
    "face_recognition_sface_2021dec.onnx": (
        f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx", 10_000_000),
}

# Two face prints with cosine similarity above this are treated as the same person
# (0.363 is the value recommended by the SFace authors). Used in Step 2.
MATCH_THRESHOLD = 0.363


@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int
    score: float
    embedding: np.ndarray  # 128 numbers, length-normalised


def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, (url, min_size) in MODELS.items():
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path) and os.path.getsize(path) >= min_size:
            continue
        print(f"Downloading {name} ...")
        tmp = path + ".tmp"
        try:
            urllib.request.urlretrieve(url, tmp)
            if os.path.getsize(tmp) < min_size:
                raise RuntimeError("downloaded file is too small")
            os.replace(tmp, path)
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise RuntimeError(
                f"Could not download {name} ({e}).\n"
                f"Download it manually from:\n  {url}\n"
                f"and put it in the '{MODEL_DIR}' folder."
            )


def shrink(img, max_side=960):
    """Make big phone photos smaller so detection is fast."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def crop_face(img, face, margin=0.4, size=200):
    """Square thumbnail around a face (for the employee list)."""
    cx, cy = face.x + face.w / 2, face.y + face.h / 2
    half = max(face.w, face.h) * (1 + margin) / 2
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(img.shape[1], cx + half)), int(min(img.shape[0], cy + half))
    return cv2.resize(img[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)


class FaceEngine:
    def __init__(self):
        ensure_models()
        self._lock = threading.Lock()  # OpenCV DNN objects are not thread-safe
        self.detector = cv2.FaceDetectorYN.create(
            os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx"),
            "", (320, 320), score_threshold=0.8, nms_threshold=0.3, top_k=20,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx"), "",
        )

    def embed_faces(self, img):
        """Find every face in a BGR image. Returns a list of Face objects."""
        with self._lock:
            h, w = img.shape[:2]
            self.detector.setInputSize((w, h))
            _, found = self.detector.detect(img)
            faces = []
            if found is None:
                return faces
            for row in found:
                aligned = self.recognizer.alignCrop(img, row)
                feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
                feat /= (np.linalg.norm(feat) + 1e-9)
                x, y, fw, fh = [int(v) for v in row[:4]]
                faces.append(Face(x, y, fw, fh, float(row[-1]), feat))
            return faces


def cosine_score(a, b):
    """Similarity between two face prints (1.0 = identical)."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
