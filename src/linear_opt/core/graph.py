r"""Graph helpers for routing models: edge indexing, components, minimum cuts.

Symmetric routing models use one variable per undirected edge
:math:`e = \{i, j\}`, :math:`i < j`, stored in row-major upper-triangular order.
:class:`EdgeIndex` maps between edges, node pairs and cut sets
:math:`\delta(S) = \{\{i, j\} : i \in S, j \notin S\}`.

Separation of subtour-elimination constraints
:math:`x(\delta(S)) \ge 2` needs a global minimum cut of the support graph
weighted by :math:`x`; :func:`min_cut_phases` implements Stoer & Wagner (1997).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from linear_opt.core.ir import FloatArray, IntArray

BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, eq=False)
class EdgeIndex:
    """Undirected edges of the complete graph on ``n`` nodes."""

    n: int
    i: IntArray = field(init=False, repr=False)
    j: IntArray = field(init=False, repr=False)
    _ids: npt.NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("need at least two nodes")
        i, j = np.triu_indices(self.n, k=1)
        ids = np.full((self.n, self.n), -1, dtype=np.int64)
        ids[i, j] = ids[j, i] = np.arange(len(i))
        object.__setattr__(self, "i", i.astype(np.int64))
        object.__setattr__(self, "j", j.astype(np.int64))
        object.__setattr__(self, "_ids", ids)

    @property
    def m(self) -> int:
        """Number of edges, ``n (n - 1) / 2``."""
        return int(self.i.shape[0])

    def id(self, a: int, b: int) -> int:
        """Index of edge ``{a, b}``."""
        if a == b:
            raise ValueError("no self-loops")
        return int(self._ids[a, b])

    def weights(self, matrix: FloatArray) -> FloatArray:
        """Per-edge values from a symmetric ``(n, n)`` matrix."""
        return np.asarray(matrix[self.i, self.j], dtype=np.float64)

    def incidence(self) -> sp.csr_array:
        """Node-edge incidence matrix ``(n, m)``: row ``v`` sums the edges at ``v``."""
        rows = np.concatenate([self.i, self.j])
        cols = np.concatenate([np.arange(self.m), np.arange(self.m)])
        return sp.csr_array((np.ones(2 * self.m), (rows, cols)), shape=(self.n, self.m))

    def delta(self, in_s: BoolArray) -> IntArray:
        """Edges with exactly one endpoint in ``S`` (boolean node mask)."""
        return np.flatnonzero(in_s[self.i] != in_s[self.j]).astype(np.int64)

    def inside(self, in_s: BoolArray) -> IntArray:
        """Edges with both endpoints in ``S``."""
        return np.flatnonzero(in_s[self.i] & in_s[self.j]).astype(np.int64)

    def to_matrix(self, x: FloatArray) -> FloatArray:
        """Symmetric ``(n, n)`` matrix from per-edge values."""
        w = np.zeros((self.n, self.n))
        w[self.i, self.j] = x
        w[self.j, self.i] = x
        return w


def components(w: FloatArray, eps: float = 1e-6) -> list[IntArray]:
    """Connected components of the graph with edges where ``w > eps``."""
    n_comp, labels = connected_components(sp.csr_array(w > eps), directed=False)
    return [np.flatnonzero(labels == k).astype(np.int64) for k in range(n_comp)]


def min_cut_phases(w: FloatArray) -> list[tuple[float, IntArray]]:
    """All "cut-of-the-phase" values of Stoer-Wagner on a dense symmetric weight matrix.

    The minimum over the returned cuts is a global minimum cut. Returning every
    phase cut lets a separator add several violated cuts per round.

    Returns:
        ``(value, nodes_on_one_side)`` for each of the ``n - 1`` phases.
    """
    n = w.shape[0]
    weights = np.array(w, dtype=np.float64, copy=True)
    np.fill_diagonal(weights, 0.0)
    groups: list[list[int]] = [[v] for v in range(n)]
    active = np.ones(n, dtype=bool)
    phases: list[tuple[float, IntArray]] = []
    while active.sum() > 1:
        nodes = np.flatnonzero(active)
        added = np.zeros(n, dtype=bool)
        key = np.zeros(n)
        prev = last = int(nodes[0])
        added[last] = True
        key += weights[last]
        for _ in range(len(nodes) - 1):
            candidates = np.where(active & ~added, key, -np.inf)
            nxt = int(np.argmax(candidates))
            prev, last = last, nxt
            cut_value = float(key[nxt])
            added[nxt] = True
            key += weights[nxt]
        phases.append((cut_value, np.array(sorted(groups[last]), dtype=np.int64)))
        weights[prev] += weights[last]
        weights[:, prev] += weights[:, last]
        weights[prev, prev] = 0.0
        weights[last] = 0.0
        weights[:, last] = 0.0
        groups[prev].extend(groups[last])
        active[last] = False
    return phases


def global_min_cut(w: FloatArray) -> tuple[float, IntArray]:
    """Value and one side of a global minimum cut (Stoer-Wagner)."""
    return min(min_cut_phases(w), key=lambda phase: phase[0])


def mask(n: int, nodes: Any) -> BoolArray:
    """Boolean membership mask of ``nodes`` in ``range(n)``."""
    out = np.zeros(n, dtype=bool)
    out[np.asarray(nodes, dtype=np.int64)] = True
    return out
