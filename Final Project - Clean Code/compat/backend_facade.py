"""
Compatibility layer that exposes the same surface as the original two-file project,
but internally routes calls to the Clean Architecture modules.
"""
from __future__ import annotations

import cv2  # noqa: F401 (used by UI)
from typing import Optional, Tuple

import numpy as np

from ..infra.persistence.sqlite_repository import SQLiteRepository as DB
from ..infra.recognition.insightface_engine import InsightFaceEngine as _Engine
from ..infra.recognition.face_index import FaceIndex as _Index
from ..usecases.attendance_runtime import AttendanceRuntime as _Runtime
from ..infra.config import (
    PROCESS_EVERY_N, SIM_THRESHOLD, DEBOUNCE_SECONDS, MIN_BOX_SIZE,  # re-export for UI
)

# Expose constants expected by the original UI
__all__ = [
    "DB", "FaceEngine", "FaceIndex", "AttendanceRuntime",
    "PROCESS_EVERY_N", "SIM_THRESHOLD", "DEBOUNCE_SECONDS", "MIN_BOX_SIZE"
]

class FaceEngine(_Engine):
    """Directly reuse the insightface adapter."""
    pass

class FaceIndex(_Index):
    """Directly reuse the index implementation."""
    pass

class AttendanceRuntime(_Runtime):
    """Same signature: (db, engine, index)."""
    pass
