from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .entities import Identity

# ---- Repository Port (Persistence) ----

class RepositoryPort(ABC):
    """Abstracts all persistence operations used by the app."""

    # Auth / Teachers
    @abstractmethod
    def create_teacher(self, username: str, password: str, full_name: str) -> int: ...
    @abstractmethod
    def get_teacher_by_username(self, username: str): ...
    @abstractmethod
    def verify_teacher(self, username: str, password: str) -> Optional[int]: ...

    # Courses / Students
    @abstractmethod
    def create_course(self, teacher_id: int, name: str, code: Optional[str] = None) -> int: ...
    @abstractmethod
    def upsert_student(self, full_name: str, student_code: Optional[str] = None) -> int: ...
    @abstractmethod
    def enroll_student(self, course_id: int, student_id: int) -> None: ...
    @abstractmethod
    def get_student(self, sid: int): ...

    # Faces
    @abstractmethod
    def add_face(self, person_type: str, person_id: int, embedding: np.ndarray,
                 quality: float | None = None, image_path: str | None = None) -> int: ...
    @abstractmethod
    def add_face_with_image(self, person_type: str, person_id: int, embedding: np.ndarray,
                            face_image_bgr: np.ndarray, quality: float | None = None) -> int: ...
    @abstractmethod
    def load_all_faces(self): ...

    # Sessions / Attendance
    @abstractmethod
    def start_session(self, course_id: int) -> int: ...
    @abstractmethod
    def end_session(self, session_id: int) -> None: ...
    @abstractmethod
    def record_event(self, session_id: int, person_type: str, person_id: int) -> None: ...
    @abstractmethod
    def export_session(self, session_id: int) -> List[Dict]: ...

    # Raw connection (for legacy UI queries). Prefer not to use in new code.
    @property
    @abstractmethod
    def conn(self): ...


# ---- Face Recognition Port ----

class FaceDetection:
    """A lightweight detection result for Clean Architecture boundary."""
    def __init__(self, bbox: np.ndarray, normed_embedding: np.ndarray):
        self.bbox = bbox
        self.normed_embedding = normed_embedding.astype("float32")

class FaceRecognitionPort(ABC):
    """Detect faces and produce normalized embeddings."""
    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> Sequence[FaceDetection]: ...


# ---- In-memory Index Port ----

class FaceIndexPort(ABC):
    @abstractmethod
    def rebuild(self, repo: RepositoryPort) -> None: ...
    @abstractmethod
    def add(self, emb: np.ndarray, idt: Identity) -> None: ...
    @abstractmethod
    def update_embedding_for_identity(self, person_type: str, person_id: int,
                                      new_emb: np.ndarray, alpha: float = 0.05) -> None: ...
    @abstractmethod
    def match(self, emb: np.ndarray, thr: float) -> tuple[Optional[Identity], float]: ...
