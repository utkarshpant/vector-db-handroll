from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
from uuid import UUID
import numpy as np

from .BaseIndex import BaseIndex

@dataclass(slots=True)
class _Ball:
    """
    A `Node` in the ball-tree, which may be a leaf or an internal node.
    - `Leaf` if both `left` and `right` are None, containing <= leaf_size points.
    - `Internal` if it has children, containing more than leaf_size points.
    """
    idx_list: np.ndarray # indices of the original vectors
    center: np.ndarray # center of the ball/node
    radius: float # radius of the ball (max cosine distance to center from points in idx_list)
    left: _Ball | None = None
    right: _Ball | None = None

class BallTreeIndex(BaseIndex):
    """
    Ball-tree with cosine distance.

    Parameters
    ----------
    leaf_size : int
        When a node holds <= `leaf_size` points, stop splitting and make it a leaf.
    """

    def __init__(self, leaf_size: int = 16) -> None:
        self.leaf_size = leaf_size
        self._vectors: np.ndarray | None = None        # (n, d) float32 unit-norm
        self._ids: list[UUID] = []                  # parallel list of UUIDs
        self._root: _Ball | None = None             # top of the tree

    def build(self, vectors: List[np.ndarray], ids: List[UUID]) -> None:
        """
        Build the ball tree in O(n log n).

        Steps
        -----
        1.  stack all vectors into one dense matrix + L2-normalize.
        2.  recursively split: (#TODO: verify recursion!!!)
              - calculate center
              - radius = max cosine distance to center
              - if no of points > leaf_size:
                    - project points onto center
                    - plit by median of that projection
        """
        if len(vectors) != len(ids):
            raise ValueError("vectors and ids length mismatch")
        if not vectors:
            self._vectors, self._ids, self._root = None, [], None
            return

        mat = np.stack([v.astype(np.float32, copy=False) for v in vectors])
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        self._vectors, self._ids = mat, list(ids)

        def build_rec(idxs: np.ndarray) -> _Ball:
            """
            Recursively build a ball tree node from the given indices.
            """
            points = mat[idxs]
            center = points.mean(axis=0)
            center /= (np.linalg.norm(center) or 1.0) # default to unit vector
            # cosine distance = 1 – dot, so radius in same units
            radius = float(np.max(1.0 - points @ center))

            node = _Ball(idx_list=idxs, center=center, radius=radius)

            # split if too many points
            if idxs.size > self.leaf_size:
                proj = points @ center
                median = np.median(proj)
                left_mask = proj <= median
                node.left = build_rec(idxs[left_mask])
                node.right = build_rec(idxs[~left_mask])
            return node

        self._root = build_rec(np.arange(mat.shape[0]))

    def search(self, query: np.ndarray, k: int) -> List[Tuple[UUID, float]]:
        """
        Return top-k nearest neighbors: (UUID, cosine_similarity).
        # TODO: consider returning chunk, similarity tuples instead?

        Pruning logic:
        --------------
        If lower_bound(query, node) >= worst_best_so_far, skip subtree.
        """
        if self._root is None or self._vectors is None:
            raise RuntimeError("Index not built")
        if k <= 0:
            return []

        # normalise query (cosine)
        q = query.astype(np.float32, copy=False)
        q /= (np.linalg.norm(q) or 1.0)

        # top-k
        best_idx: list[int] = []
        best_dst: list[float] = []

        def push(i: int, dist: float) -> None:
            """Insert into heap."""
            if len(best_idx) < k:
                best_idx.append(i); best_dst.append(dist)
            elif dist < max(best_dst):
                worst = best_dst.index(max(best_dst))
                best_idx[worst] = i; best_dst[worst] = dist

        def visit(node: _Ball) -> None:
            """DFS with pruning."""
            lb = max(0.0, 1.0 - float(q @ node.center) - node.radius)
            if len(best_dst) == k and lb >= max(best_dst):
                return  # nothing inside can beat current worst;

            if node.left is None and node.right is None:
                for i in node.idx_list:
                    d = 1.0 - float(q @ self._vectors[i])
                    push(int(i), d)
                return

            l_dist = 1.0 - float(q @ node.left.center) if node.left else float("inf")
            r_dist = 1.0 - float(q @ node.right.center) if node.right else float("inf")
            first, second = (node.left, node.right) if l_dist < r_dist else (node.right, node.left)
            visit(first)
            if second:
                visit(second)

        visit(self._root)

        # convert to similarity and sort in desc order
        order = np.argsort(best_dst)
        return [(self._ids[best_idx[i]], 1.0 - best_dst[i]) for i in order]

    def to_string(self) -> str:
        """
        Return a string representation of the index; debugging util
        """
        def _to_str(node: _Ball, depth: int = 0) -> str:
            indent = "  " * depth
            if node.left is None and node.right is None:
                return f"{indent}Leaf: {len(node.idx_list)} points, center={node.center}, radius={node.radius}\n"
            left_str = _to_str(node.left, depth + 1) if node.left else ""
            right_str = _to_str(node.right, depth + 1) if node.right else ""
            return f"{indent}Node: {len(node.idx_list)} points, center={node.center}, radius={node.radius}\n{left_str}{right_str}"

        return _to_str(self._root) if self._root else "Empty BallTreeIndex"