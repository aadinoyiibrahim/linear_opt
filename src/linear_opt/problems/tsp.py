r"""Symmetric travelling salesman problem (MILP) in three formulations.

* **DFJ** (Dantzig-Fulkerson-Johnson, 1954): one binary per edge, degree 2 at
  every node, and the exponentially many subtour-elimination constraints
  :math:`x(\delta(S)) \ge 2` added *lazily* when a candidate solution violates
  one. Separation is exact: connected components for integer points, a
  Stoer-Wagner minimum cut for fractional points. Its LP relaxation is the
  Held-Karp bound, typically within ~1 % of the optimum.
* **MTZ** (Miller-Tucker-Zemlin, 1960): arcs plus order variables
  :math:`u_i - u_j + (n-1) x_{ij} \le n-2`. Polynomial size, very weak LP.
* **SCF** (single-commodity flow, Gavish-Graves 1978): the depot ships
  :math:`n-1` units, every other city consumes one; flow only on used arcs.
  Polynomial size, LP between MTZ and DFJ.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel, ConfigDict

from linear_opt.core.config import ProblemKind
from linear_opt.core.graph import EdgeIndex, components, mask, min_cut_phases
from linear_opt.core.ir import (
    FloatArray,
    IntArray,
    ModelBuilder,
    Sense,
    VarBlock,
    VarType,
    sum_over,
)
from linear_opt.core.model import OptimizationModel
from linear_opt.core.result import Cut, LazyOracle, ProgressPoint, RawSolution
from linear_opt.data.distances import haversine_matrix
from linear_opt.data.geonames import City
from linear_opt.data.tsplib import TsplibProblem, tour_length
from linear_opt.problems.heuristics import nearest_neighbour, two_opt

EPS = 1e-6
MAX_HISTORY = 200


@dataclass(frozen=True, eq=False)
class TSPInstance:
    """Immutable symmetric TSP instance."""

    names: tuple[str, ...]
    dist: FloatArray
    coords: FloatArray | None = None
    geographic: bool = False
    reference: float | None = None
    optimal_tour: tuple[int, ...] | None = None
    unit: str = "distance"

    def __post_init__(self) -> None:
        n = len(self.names)
        if n < 3:
            raise ValueError("a TSP needs at least 3 cities")
        if self.dist.shape != (n, n):
            raise ValueError(f"dist has shape {self.dist.shape}, expected {(n, n)}")
        if not np.allclose(self.dist, self.dist.T):
            raise ValueError("distance matrix must be symmetric")
        if np.any(self.dist < 0) or not np.all(np.isfinite(self.dist)):
            raise ValueError("distances must be finite and non-negative")
        if self.coords is not None and self.coords.shape != (n, 2):
            raise ValueError(f"coords has shape {self.coords.shape}, expected {(n, 2)}")

    @property
    def n(self) -> int:
        """Number of cities."""
        return len(self.names)

    @classmethod
    def from_tsplib(
        cls,
        problem: TsplibProblem,
        reference: float | None = None,
        tour: Sequence[int] | None = None,
    ) -> TSPInstance:
        """Wrap a parsed TSPLIB problem (coordinates are planar or display-only)."""
        return cls(
            tuple(str(v + 1) for v in range(problem.dimension)),
            problem.distances,
            problem.coords,
            geographic=False,
            reference=reference,
            optimal_tour=tuple(tour) if tour is not None else None,
        )

    @classmethod
    def random(cls, n: int, *, seed: int = 0) -> TSPInstance:
        """Uniform points in a 1000 x 1000 square, rounded Euclidean distances."""
        pts = np.random.default_rng(seed).random((n, 2)) * 1000
        dist = np.floor(np.linalg.norm(pts[:, None] - pts[None, :], axis=2) + 0.5)
        return cls(tuple(f"N{i}" for i in range(n)), dist, pts)

    @classmethod
    def from_cities(cls, cities: Sequence[City]) -> TSPInstance:
        """Round trip through German cities; distances in whole km (great circle)."""
        xy = np.array([[c.lat, c.lon] for c in cities])
        dist = np.floor(haversine_matrix(xy, xy) + 0.5)
        return cls(tuple(c.name for c in cities), dist, xy, geographic=True, unit="km")


class TSPOptions(BaseModel):
    """Options in the ``[model]`` table of a TSP config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    formulation: Literal["dfj", "mtz", "scf"] = "dfj"
    warm_start: Literal["two_opt", "none"] = "two_opt"
    root_cuts: bool = True  # fractional subtour cuts at the root (Gurobi user cuts)


@dataclass(frozen=True)
class SubtourFrame:
    """One integer candidate seen by the separator: its edges and subtours."""

    edges: tuple[tuple[int, int], ...]
    n_subtours: int


@dataclass(frozen=True, eq=False)
class TSPSolution:
    """Optimal tour, its length and the lazy-constraint history."""

    tour: tuple[int, ...]
    length: float
    objective: float
    history: tuple[SubtourFrame, ...] = ()
    progress: tuple[ProgressPoint, ...] = ()

    def edges(self) -> list[tuple[int, int]]:
        """Undirected tour edges ``(a, b)`` with ``a < b``."""
        t = self.tour
        return [tuple(sorted((t[k], t[(k + 1) % len(t)]))) for k in range(len(t))]  # type: ignore[misc]


def tour_from_successors(succ: dict[int, int], start: int = 0) -> list[int]:
    """Follow successor pointers from ``start`` until the cycle closes."""
    tour = [start]
    while (nxt := succ[tour[-1]]) != start:
        if nxt in tour or len(tour) > len(succ):
            raise ValueError("successor map does not describe a single tour")
        tour.append(nxt)
    if len(tour) != len(succ):
        raise ValueError(
            f"successor map does not describe a single tour ({len(tour)} of {len(succ)})"
        )
    return tour


def tour_from_edges(n: int, edges: Sequence[tuple[int, int]]) -> list[int]:
    """Hamiltonian cycle from its undirected edge list."""
    adj: dict[int, list[int]] = {v: [] for v in range(n)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    tour, prev = [0], -1
    while len(tour) < n:
        cur = tour[-1]
        nxt = next((v for v in adj[cur] if v != prev), None)
        if nxt is None or nxt == 0:
            raise ValueError("edges do not form a Hamiltonian cycle")
        prev = cur
        tour.append(nxt)
    return tour


class TSPModel(OptimizationModel[TSPInstance, TSPSolution, TSPOptions]):
    """TSP on a :class:`TSPInstance`."""

    kind: ClassVar[ProblemKind] = ProblemKind.TSP
    options_type: ClassVar[type[BaseModel]] = TSPOptions

    _x: VarBlock
    _aux: VarBlock | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.edge_index = EdgeIndex(self.data.n)
        self.history: list[SubtourFrame] = []

    # ------------------------------------------------------------------ build
    def _build(self, builder: ModelBuilder) -> None:
        n, d = self.data.n, self.data.dist
        self._aux = None
        if self.options.formulation == "dfj":
            e = self.edge_index
            self._x = builder.add_vars("edge", e.m, vtype=VarType.BINARY, obj=e.weights(d))
            builder.add_constrs("degree", [(self._x, e.incidence())], Sense.EQ, 2.0)
            return

        ub = 1.0 - np.eye(n)  # no self-loops
        self._x = builder.add_vars("arc", (n, n), ub=ub, vtype=VarType.BINARY, obj=d)
        builder.add_constrs("leave", [(self._x, sum_over((n, n), axis=1))], Sense.EQ, 1.0)
        builder.add_constrs("enter", [(self._x, sum_over((n, n), axis=0))], Sense.EQ, 1.0)
        i, j = np.nonzero(~np.eye(n, dtype=bool))
        rows = np.arange(len(i))
        if self.options.formulation == "mtz":
            lb = np.r_[0.0, np.ones(n - 1)]
            ubu = np.r_[0.0, np.full(n - 1, n - 1.0)]
            self._aux = builder.add_vars("order", n, lb=lb, ub=ubu)
            keep = (i > 0) & (j > 0)
            i, j, rows = i[keep], j[keep], np.arange(int(keep.sum()))
            arc = sp.csr_array(
                ((n - 1.0) * np.ones(len(i)), (rows, i * n + j)), shape=(len(i), n * n)
            )
            order = sp.csr_array(
                (np.r_[np.ones(len(i)), -np.ones(len(i))], (np.r_[rows, rows], np.r_[i, j])),
                shape=(len(i), n),
            )
            builder.add_constrs("mtz", [(self._x, arc), (self._aux, order)], Sense.LE, n - 2.0)
        else:  # scf
            self._aux = builder.add_vars("flow", (n, n), ub=(n - 1.0) * ub)
            inflow, outflow = sum_over((n, n), axis=0), sum_over((n, n), axis=1)
            supply = np.r_[-(n - 1.0), np.ones(n - 1)]
            builder.add_constrs("conserve", [(self._aux, inflow - outflow)], Sense.EQ, supply)
            pick = sp.csr_array((np.ones(len(i)), (rows, i * n + j)), shape=(len(i), n * n))
            builder.add_constrs(
                "carry", [(self._aux, pick), (self._x, -(n - 1.0) * pick)], Sense.LE, 0.0
            )

    # ------------------------------------------------------------ separation
    def separate(self, x: FloatArray, *, record: bool = True) -> list[Cut]:
        """Violated subtour cuts ``x(delta(S)) >= 2`` for an edge vector ``x``."""
        e, n = self.edge_index, self.data.n
        values = x[self._x.slice]
        w = e.to_matrix(values)
        comps = components(w, EPS)
        integral = bool(np.all(np.minimum(values, 1 - values) < EPS))
        if record and integral and len(self.history) < MAX_HISTORY:
            chosen = np.flatnonzero(values > 0.5)
            edges = tuple((int(e.i[k]), int(e.j[k])) for k in chosen)
            self.history.append(SubtourFrame(edges, len(comps)))
        sets: list[IntArray] = []
        if len(comps) > 1:
            sets = comps if len(comps) > 2 else comps[:1]  # S and its complement give one cut
        elif not integral:
            seen: set[frozenset[int]] = set()
            for value, side in min_cut_phases(w):
                key = (
                    frozenset(side.tolist())
                    if 0 not in side
                    else frozenset(set(range(n)) - set(side.tolist()))
                )
                if value < 2.0 - EPS and key not in seen:
                    seen.add(key)
                    sets.append(side)
        return [
            Cut(cols := self._x.offset + e.delta(mask(n, s)), np.ones(len(cols)), Sense.GE, 2.0)
            for s in sets[:50]
        ]

    def lazy_constraints(self) -> LazyOracle | None:
        """Subtour elimination for DFJ (exact on integer and fractional points)."""
        return self.separate if self.options.formulation == "dfj" else None

    def user_cuts(self) -> LazyOracle | None:
        """Fractional subtour cuts at the root node (DFJ only)."""
        if self.options.formulation != "dfj" or not self.options.root_cuts:
            return None
        return lambda x: self.separate(x, record=False)

    # ------------------------------------------------------------- warm start
    def heuristic_tour(self) -> list[int]:
        """Nearest neighbour followed by 2-opt."""
        return two_opt(self.data.dist, nearest_neighbour(self.data.dist))

    def warm_start(self) -> FloatArray | None:
        """MIP start from :meth:`heuristic_tour` in the active formulation's variables."""
        if self.options.warm_start == "none":
            return None
        n, tour = self.data.n, self.heuristic_tour()
        start = np.zeros(self.build().num_vars)
        succ = {tour[k]: tour[(k + 1) % n] for k in range(n)}
        if self.options.formulation == "dfj":
            for a, b in succ.items():
                start[self._x.offset + self.edge_index.id(a, b)] = 1.0
            return start
        position = {v: k for k, v in enumerate(tour)}
        for a, b in succ.items():
            start[self._x.offset + a * n + b] = 1.0
        assert self._aux is not None
        if self.options.formulation == "mtz":
            for v in range(n):
                start[self._aux.offset + v] = position[v]
        else:
            for a, b in succ.items():
                start[self._aux.offset + a * n + b] = n - 1 - position[a]
        return start

    def reference_objective(self) -> float | None:
        """Published optimum (TSPLIB), if known."""
        return self.data.reference

    @classmethod
    def variants(cls) -> dict[str, dict[str, Any]]:
        """Formulations compared by ``linopt variants``."""
        return {
            "dfj (lazy)": {"formulation": "dfj"},
            "single-commodity flow": {"formulation": "scf"},
            "mtz": {"formulation": "mtz"},
        }

    # ------------------------------------------------------------- solution
    def _extract(self, raw: RawSolution) -> TSPSolution:
        assert raw.x is not None
        assert raw.objective is not None
        n = self.data.n
        values = raw.x[self._x.slice]
        if self.options.formulation == "dfj":
            e = self.edge_index
            chosen = np.flatnonzero(values > 0.5)
            tour = tour_from_edges(n, [(int(e.i[k]), int(e.j[k])) for k in chosen])
        else:
            arcs = values.reshape(n, n) > 0.5
            tour = tour_from_successors({int(a): int(np.argmax(arcs[a])) for a in range(n)})
        return TSPSolution(
            tour=tuple(tour),
            length=tour_length(self.data.dist, tour),
            objective=raw.objective,
            history=tuple(self.history),
            progress=raw.progress,
        )

    def validate(self, solution: TSPSolution) -> list[str]:
        """Tour visits every city exactly once and its length matches the objective."""
        issues = []
        if sorted(solution.tour) != list(range(self.data.n)):
            issues.append("tour is not a permutation of all cities")
        if not np.isclose(solution.length, solution.objective, rtol=1e-6, atol=1e-6):
            issues.append(
                f"objective {solution.objective:.6f} != tour length {solution.length:.6f}"
            )
        return issues
