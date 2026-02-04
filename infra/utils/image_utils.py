from __future__ import annotations
from datetime import datetime
from pathlib import Path
import time
from typing import Tuple
import cv2
import numpy as np

from ..config import IMAGES_DIR, MIN_BLUR_VAR

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def l2_normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    n = np.sqrt((v * v).sum()) + eps
    return v / n

def is_blurry(img_bgr: np.ndarray, thr: float = MIN_BLUR_VAR) -> bool:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < thr

def crop_from_bbox(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    l, t, r, b = bbox
    h, w = frame.shape[:2]
    l = max(0, l); t = max(0, t)
    r = min(w, r); b = min(h, b)
    return frame[t:b, l:r]

def save_face_image(person_type: str, person_id: int, img_bgr: np.ndarray) -> str:
    ts = int(time.time())
    filename = IMAGES_DIR / f"{person_type}_{person_id}_{ts}.jpg"
    try:
        cv2.imwrite(str(filename), img_bgr)
    except Exception:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(filename), img_bgr)
    return str(filename)
