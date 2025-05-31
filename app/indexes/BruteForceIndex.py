from __future__ import annotations
from typing import List
from uuid import UUID, uuid4
from .BaseIndex import BaseIndex
import numpy as np

class BruteForceIndex(BaseIndex):
    """
    A super simple KNN index: store every vector and scan all at query timr.
    """

    name = "Brute-force Index"

    def __init__(self, normalize: bool = True):
        self._vectors: np.ndarray | None = None
        self._norms: np.ndarray | None = None
        self._ids: List[UUID] = []
        self._normalize = normalize

    def build(self, vectors: List[List[float]], ids: List[UUID]) -> None:
        if len(vectors) != len(ids):
            raise ValueError("Vectors and IDs must have the same length")
        if not vectors:
            self._vectors = np.empty([0, 0], dtype=np.float32).tolist()
            self._norms = np.empty([0, 1], dtype=np.float32).tolist()
            self._ids = []
            return
        
        # (n, d) float32 matrix – copy=False avoids dup if already np.ndarray
        mat = np.stack([np.array(v).astype(np.float32, copy=False) for v in vectors])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid divide-by-zero later

        if self._normalize:
            mat /= norms          # in-place; now every row has unit norm
            norms = np.ones_like(norms)

        self._vectors, self._norms, self._ids = mat, norms, list(ids)

    def search(self, query: List[float], k: int) -> List[tuple[UUID, float]]:
        if self._vectors is None:
            raise RuntimeError("Index has not been built yet")
        if k <= 0:
            raise ValueError("k must be a positive integer")
        q = np.array(query).astype(np.float32, copy=False)
        if (self._normalize):
            q /= (np.linalg.norm(query) or 1.0)
            similarities = np.dot(self._vectors, query)
        else:
            qnorm = (np.linalg.norm(query) or 1.0)
            if self._norms is None:
                raise RuntimeError("Index norms have not been computed. Build the index first.")
            similarities = np.dot(self._vectors, query) / (self._norms.squeeze() * qnorm)

        k = min(k, similarities.size)
        idx_unsorted = np.argpartition(-similarities, k - 1)[:k]
        idx_sorted   = idx_unsorted[np.argsort(-similarities[idx_unsorted])]

        return [(self._ids[i], float(similarities[i])) for i in idx_sorted]