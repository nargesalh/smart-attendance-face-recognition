from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from ...core.entities import Identity
from ...core.ports import FaceIndexPort, RepositoryPort
from ..utils.image_utils import l2_normalize

class FaceIndex(FaceIndexPort):
    def __init__(self):
        self.embs: List[np.ndarray] = []
        self.ids: List[Identity] = []

    def rebuild(self, repo: RepositoryPort) -> None:
        self.embs.clear(); self.ids.clear()
        cur = repo.conn.cursor()
        cur.execute("""
            SELECT f.person_type, f.person_id, f.embedding,
                   COALESCE(s.full_name, t.full_name) as full_name,
                   s.student_code as code
            FROM faces f
            LEFT JOIN students s ON (f.person_type='student' AND s.id=f.person_id)
            LEFT JOIN teachers t ON (f.person_type='teacher' AND t.id=f.person_id)
        """)
        for r in cur.fetchall():
            idt = Identity(r["person_type"], int(r["person_id"]), r["full_name"], r["code"])
            emb = np.frombuffer(r["embedding"], dtype=np.float32)
            self.add(emb, idt)

    def add(self, emb: np.ndarray, idt: Identity) -> None:
        self.embs.append(l2_normalize(emb.astype(np.float32)))
        self.ids.append(idt)

    def update_embedding_for_identity(self, person_type: str, person_id: int, new_emb: np.ndarray, alpha: float = 0.05):
        new_emb = l2_normalize(new_emb.astype(np.float32))
        for i, idt in enumerate(self.ids):
            if idt.person_type == person_type and idt.person_id == person_id:
                base = self.embs[i]
                updated = (1 - alpha) * base + alpha * new_emb
                self.embs[i] = l2_normalize(updated)
                break

    def match(self, emb: np.ndarray, thr: float) -> Tuple[Optional[Identity], float]:
        if not self.embs:
            return None, -1.0
        emb = l2_normalize(emb.astype(np.float32))
        sims = np.dot(np.stack(self.embs, axis=0), emb)
        k = int(np.argmax(sims))
        best = float(sims[k])
        if best >= thr:
            return self.ids[k], best
        return None, best
