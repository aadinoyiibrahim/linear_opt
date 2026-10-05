"""Solver-independent solve results and lazy-constraint types."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from linear_opt.core.ir import FloatArray, IntArray, Sense


class SolveStatus(StrEnum):
    """Termination status, harmonised across backends."""

    OPTIMAL = "optimal"
    TIME_LIMIT = "time_limit"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    INF_OR_UNBD = "infeasible_or_unbounded"
    INTERRUPTED = "interrupted"
    NUMERIC = "numeric_error"
    OTHER = "other"


@dataclass(frozen=True)
class ProgressPoint:
    """Incumbent and best bound at a moment of a MIP solve."""

    time: float
    incumbent: float | None
    bound: float | None


@dataclass(frozen=True, eq=False)
class Cut:
    """A single linear constraint ``sum(coefs * x[cols]) sense rhs`` added during the solve."""

    cols: IntArray
    coefs: FloatArray
    sense: Sense
    rhs: float

    def violation(self, x: FloatArray) -> float:
        """Amount by which ``x`` violates the cut (``<= 0`` means satisfied)."""
        lhs = float(np.dot(self.coefs, x[self.cols]))
        if self.sense is Sense.LE:
            return lhs - self.rhs
        if self.sense is Sense.GE:
            return self.rhs - lhs
        return abs(lhs - self.rhs)


class ProgressRecorder:
    """Collect incumbent/bound changes from a solver callback (deduplicated).

    Values whose magnitude is at least ``infinity`` (e.g. ``GRB.INFINITY`` or
    ``inf``) are stored as ``None`` - "no incumbent yet" / "no bound yet".
    """

    def __init__(self, infinity: float = float("inf")) -> None:
        self.infinity = infinity
        self.points: list[ProgressPoint] = []

    def record(self, t: float, incumbent: float, bound: float) -> None:
        """Append a point if incumbent or bound changed."""
        inc = None if not abs(incumbent) < self.infinity else float(incumbent)
        bnd = None if not abs(bound) < self.infinity else float(bound)
        last = self.points[-1] if self.points else None
        if last is None or (last.incumbent, last.bound) != (inc, bnd):
            self.points.append(ProgressPoint(float(t), inc, bnd))


#: Separation oracle: given an integer-feasible ``x``, return violated cuts (empty if none).
LazyOracle = Callable[[FloatArray], Sequence[Cut]]


@dataclass(frozen=True, eq=False)
class RawSolution:
    """What a backend returns: status, primal/dual vectors and statistics.

    Dual sign convention (shared by Gurobi and HiGHS, checked in the tests):
    for a minimisation, duals of ``>=`` rows are non-negative and of ``<=`` rows
    non-positive; signs flip for maximisation. Reduced costs are
    ``c - A^T y``.
    """

    status: SolveStatus
    backend: str
    backend_version: str
    objective: float | None = None
    bound: float | None = None
    x: FloatArray | None = None
    duals: FloatArray | None = None
    reduced_costs: FloatArray | None = None
    runtime: float = 0.0
    iterations: int = 0
    nodes: int = 0
    progress: tuple[ProgressPoint, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_solution(self) -> bool:
        """Whether a feasible primal solution is available."""
        return self.x is not None

    @property
    def gap(self) -> float | None:
        """Relative optimality gap ``|obj - bound| / max(|obj|, 1e-10)``."""
        if self.objective is None or self.bound is None:
            return None
        return abs(self.objective - self.bound) / max(abs(self.objective), 1e-10)
