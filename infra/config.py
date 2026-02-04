from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple

# Base directories (12-factor: allow override via env vars)
BASE_DATA_DIR = Path(os.getenv("ATT_DATA_DIR", "./data")).resolve()
IMAGES_DIR = Path(os.getenv("ATT_IMAGES_DIR", str(BASE_DATA_DIR / "face_images"))).resolve()
DB_PATH = Path(os.getenv("ATT_DB_PATH", str(BASE_DATA_DIR / "attendance.db"))).resolve()

# Make sure directories exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Face / recog params
INSIGHT_ROOT = os.getenv("INSIGHTFACE_ROOT", str(Path.home() / ".insightface"))
DETECT_SIZE: Tuple[int, int] = (640, 640)

# Runtime tuning
PROCESS_EVERY_N = int(os.getenv("ATT_PROCESS_EVERY_N", "1"))
SIM_THRESHOLD = float(os.getenv("ATT_SIM_THRESHOLD", "0.58"))
DEBOUNCE_SECONDS = float(os.getenv("ATT_DEBOUNCE_SECONDS", "10"))
MIN_BOX_SIZE = int(os.getenv("ATT_MIN_BOX_SIZE", "80"))
MIN_BLUR_VAR = float(os.getenv("ATT_MIN_BLUR_VAR", "20.0"))
