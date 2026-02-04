from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

# ---- Domain Entities & Value Objects ----

@dataclass(frozen=True)
class Identity:
    person_type: str  # 'student' | 'teacher'
    person_id: int
    name: str
    code: Optional[str] = None

@dataclass
class Teacher:
    id: int
    username: str
    full_name: str

@dataclass
class Course:
    id: int
    teacher_id: int
    name: str
    code: Optional[str] = None

@dataclass
class Student:
    id: int
    full_name: str
    student_code: Optional[str] = None

@dataclass
class FaceRecord:
    id: int
    person_type: str
    person_id: int
    embedding: np.ndarray
    quality: Optional[float] = None
    image_path: Optional[str] = None

@dataclass
class Session:
    id: int
    course_id: int
    started_at: str
    ended_at: Optional[str]

@dataclass
class AttendanceRow:
    session_id: int
    person_type: str
    person_id: int
    first_seen: str
    last_seen: str

@dataclass
class Event:
    id: int
    session_id: int
    person_type: str
    person_id: int
    ts: str
