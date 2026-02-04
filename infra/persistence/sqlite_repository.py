from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from passlib.hash import bcrypt

from ...core.ports import RepositoryPort
from ..config import DB_PATH
from ..utils.image_utils import now_ts, l2_normalize, save_face_image

class SQLiteRepository(RepositoryPort):
    """SQLite implementation of RepositoryPort. Safe for single-process desktop apps."""
    def __init__(self, db_path: Path = DB_PATH):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # --- public property for legacy UI access ---
    @property
    def conn(self):
        return self._conn

    # --- schema ---
    def _init_schema(self):
        cur = self._conn.cursor()
        cur.executescript("""
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id INTEGER NOT NULL REFERENCES teachers(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            code TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            student_code TEXT UNIQUE,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS enrollments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
            student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
            UNIQUE(course_id, student_id)
        );

        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_type TEXT NOT NULL,
            person_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            quality REAL,
            image_path TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
            started_at TEXT NOT NULL,
            ended_at TEXT
        );

        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            person_type TEXT NOT NULL,
            person_id INTEGER NOT NULL,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            UNIQUE(session_id, person_type, person_id)
        );

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            person_type TEXT NOT NULL,
            person_id INTEGER NOT NULL,
            ts TEXT NOT NULL
        );
        """)
        self._conn.commit()

    # --- teachers/auth ---
    def create_teacher(self, username: str, password: str, full_name: str) -> int:
        cur = self._conn.cursor()
        pwh = bcrypt.hash(password)
        cur.execute(
            "INSERT INTO teachers(username, password_hash, full_name, created_at) VALUES(?,?,?,?)",
            (username, pwh, full_name, now_ts()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_teacher_by_username(self, username: str):
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM teachers WHERE username=?", (username,))
        return cur.fetchone()

    def verify_teacher(self, username: str, password: str) -> Optional[int]:
        row = self.get_teacher_by_username(username)
        if not row:
            return None
        try:
            ok = bcrypt.verify(password, row["password_hash"])
        except Exception:
            ok = False
        return int(row["id"]) if ok else None

    # --- courses/students ---
    def create_course(self, teacher_id: int, name: str, code: Optional[str] = None) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO courses(teacher_id, name, code, created_at) VALUES(?,?,?,?)",
            (teacher_id, name, code, now_ts()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def upsert_student(self, full_name: str, student_code: Optional[str] = None) -> int:
        cur = self._conn.cursor()
        if student_code:
            cur.execute("SELECT id FROM students WHERE student_code=?", (student_code,))
            r = cur.fetchone()
            if r:
                cur.execute("UPDATE students SET full_name=? WHERE id=?", (full_name, r["id"]))
                self._conn.commit()
                return int(r["id"])
        cur.execute(
            "INSERT INTO students(full_name, student_code, created_at) VALUES(?,?,?)",
            (full_name, student_code, now_ts()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def enroll_student(self, course_id: int, student_id: int) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO enrollments(course_id, student_id) VALUES(?,?)",
            (course_id, student_id),
        )
        self._conn.commit()

    def get_student(self, sid: int):
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM students WHERE id=?", (sid,))
        return cur.fetchone()

    # --- faces ---
    def add_face(self, person_type: str, person_id: int, embedding: np.ndarray,
                 quality: float | None = None, image_path: str | None = None) -> int:
        cur = self._conn.cursor()
        emb = l2_normalize(embedding.astype(np.float32))
        cur.execute(
            "INSERT INTO faces(person_type, person_id, embedding, quality, image_path, created_at) "
            "VALUES(?,?,?,?,?,?)",
            (person_type, person_id, emb.tobytes(), quality, image_path, now_ts()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def add_face_with_image(self, person_type: str, person_id: int, embedding: np.ndarray,
                            face_image_bgr: np.ndarray, quality: float | None = None) -> int:
        img_path = save_face_image(person_type, person_id, face_image_bgr)
        return self.add_face(person_type, person_id, embedding, quality=quality, image_path=img_path)

    def load_all_faces(self):
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM faces")
        return cur.fetchall()

    # --- sessions/attendance ---
    def start_session(self, course_id: int) -> int:
        cur = self._conn.cursor()
        cur.execute("INSERT INTO sessions(course_id, started_at) VALUES(?,?)", (course_id, now_ts()))
        self._conn.commit()
        return int(cur.lastrowid)

    def end_session(self, session_id: int) -> None:
        cur = self._conn.cursor()
        cur.execute("UPDATE sessions SET ended_at=? WHERE id=?", (now_ts(), session_id))
        self._conn.commit()

    def record_event(self, session_id: int, person_type: str, person_id: int) -> None:
        ts = now_ts()
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO events(session_id, person_type, person_id, ts) VALUES(?,?,?,?)",
            (session_id, person_type, person_id, ts),
        )
        cur.execute(
            "SELECT id FROM attendance WHERE session_id=? AND person_type=? AND person_id=?",
            (session_id, person_type, person_id),
        )
        r = cur.fetchone()
        if r:
            cur.execute("UPDATE attendance SET last_seen=? WHERE id=?", (ts, r["id"]))
        else:
            cur.execute(
                "INSERT INTO attendance(session_id, person_type, person_id, first_seen, last_seen) "
                "VALUES(?,?,?,?,?)",
                (session_id, person_type, person_id, ts, ts),
            )
        self._conn.commit()

    def export_session(self, session_id: int) -> List[Dict]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT a.person_type, a.person_id, a.first_seen, a.last_seen FROM attendance a WHERE a.session_id=?",
            (session_id,),
        )
        return [dict(r) for r in cur.fetchall()]
