"""Local embeddings using sentence-transformers."""

from __future__ import annotations

import numpy as np


class LocalEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = None
        self.model_name = model_name

    def encode(self, text: str) -> list[float]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model.encode([text])[0].tolist()

    def similarity(self, emb1: list[float], emb2: list[float]) -> float:
        v1, v2 = np.array(emb1), np.array(emb2)
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            return 0.0
        return float(np.dot(v1, v2) / denom)
