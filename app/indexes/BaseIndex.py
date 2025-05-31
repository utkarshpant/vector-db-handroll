# app/index/base.py

from abc import ABC, abstractmethod
from typing import List, Tuple
from uuid import UUID

import numpy as np

class BaseIndex(ABC):
    """
    Abstract base class for vector indexes.

    Concrete implementations must provide:
      - build: ingest a collection of vectors and their identifiers.
      - search: return top-k nearest neighbors for a query vector.
    """

    name: str
    """
    Name of the index implementation, used for identification.
    """

    @abstractmethod
    def build(self, vectors: List[List[float]], ids: List[UUID]) -> None:
        """
        Build or rebuild the index from scratch.

        :param vectors: list of numpy arrays representing embeddings
        :param ids: list of UUIDs corresponding to each embedding
        """
        ...

    @abstractmethod
    def search(self, query: List[float], k: int) -> List[Tuple[UUID, float]]:
        """
        Query the index to find the k most similar vectors.

        :param query: numpy array representing the query embedding
        :param k: number of nearest neighbors to return
        :return: list of (UUID, similarity_score) tuples sorted by score descending
        """
        ...

