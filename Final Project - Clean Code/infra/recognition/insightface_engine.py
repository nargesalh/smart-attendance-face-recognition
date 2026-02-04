from __future__ import annotations
from typing import Sequence

import numpy as np
from insightface.app import FaceAnalysis

from ...core.ports import FaceRecognitionPort, FaceDetection
from ..config import DETECT_SIZE, INSIGHT_ROOT

class InsightFaceEngine(FaceRecognitionPort):
    """Adapter for InsightFace FaceAnalysis to our FaceRecognitionPort."""
    def __init__(self, providers: list[str] | None = None):
        providers = providers or ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", root=INSIGHT_ROOT, providers=providers)
        # ctx_id=0 means CPU for onnxruntime; for CUDA set providers accordingly
        self.app.prepare(ctx_id=0, det_size=DETECT_SIZE)

    def detect(self, frame_bgr: np.ndarray) -> Sequence[FaceDetection]:
        faces = self.app.get(frame_bgr)
        outs: list[FaceDetection] = []
        for f in faces:
            bbox = f.bbox.astype(int)
            emb = f.normed_embedding.astype("float32")
            outs.append(FaceDetection(bbox=bbox, normed_embedding=emb))
        return outs
