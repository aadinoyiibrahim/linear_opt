"""Backend interface, factory and generic row generation."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import replace
from enum import StrEnum
from typing import ClassVar

from linear_opt.core.backends import resolve_backend
from linear_opt.core.config import BackendName, SolverSettings
from linear_opt.core.ir import FloatArray, LinearModel
from linear_opt.core.result import Cut, LazyOracle, RawSolution, SolveStatus


class Capability(StrEnum):
    """Optional features a backend may support."""

    DUALS = "duals"  # LP duals and reduced costs
    LAZY_IN_TREE = "lazy_in_tree"  # lazy constraints inside branch-and-bound
    PROGRESS = "progress"  # incumbent / bound trajectory during MIP solves
    WARM_START = "warm_start"  # MIP start
    IIS = "iis"  # irreducible infeasible subsystem
    USER_CUTS = "user_cuts"  # cuts separated at fractional nodes
    EXPORT = "export"  # write .lp / .mps


class LicenseLimitError(RuntimeError):
    """The model exceeds the size limit of the installed solver licence."""


class SolverBackend(ABC):
    """Translate a :class:`LinearModel`, solve it, and harmonise the result.

    Every backend accepts a lazy-constraint oracle. Backends with
    :attr:`Capability.LAZY_IN_TREE` call it inside branch-and-bound; the
    others fall back to *row generation* (solve, separate, add cuts, re-solve),
    which is slower but returns the same optimum.
    """

    name: ClassVar[BackendName]
    capabilities: ClassVar[frozenset[Capability]]

    @abstractmethod
    def version(self) -> str:
        """Version string of the underlying solver."""

    @abstractmethod
    def solve(
        self,
        model: LinearModel,
        settings: SolverSettings,
        *,
        lazy: LazyOracle | None = None,
        warm_start: FloatArray | None = None,
        user_cuts: LazyOracle | None = None,
    ) -> RawSolution:
        """Solve ``model``.

        Args:
            model: The model in canonical form.
            settings: Time limit, gap, threads, verbosity, export path.
            lazy: Optional separation oracle for lazy constraints. For LPs it is
                applied by row generation, so ``model.relaxed()`` plus the oracle
                gives the LP bound of the *full* formulation.
            warm_start: Optional starting point (``NaN`` entries are left free).
            user_cuts: Optional separator for fractional points (backends
                without :attr:`Capability.USER_CUTS` ignore it).

        Raises:
            LicenseLimitError: If the licence does not permit a model this large.
        """

    def write_model(self, model: LinearModel, path: str) -> None:
        """Write ``model`` without solving (format from the suffix, e.g. ``.lp``, ``.mps``)."""
        raise NotImplementedError(f"{self.name} cannot export models")

    def compute_iis(self, model: LinearModel, settings: SolverSettings) -> list[str]:
        """Names of constraints/bounds in an irreducible infeasible subsystem."""
        raise NotImplementedError(f"{self.name} does not support IIS computation")

    def supports(self, capability: Capability) -> bool:
        """Whether this backend provides ``capability``."""
        return capability in self.capabilities


def solve_by_row_generation(
    solve: Callable[[LinearModel, SolverSettings], RawSolution],
    model: LinearModel,
    settings: SolverSettings,
    oracle: LazyOracle,
    max_rounds: int = 10_000,
) -> RawSolution:
    """Solve, separate, add violated cuts, re-solve - until none is violated.

    Works for LPs and MIPs on any backend. Returned duals cover the original
    rows only.
    """
    t0 = time.perf_counter()
    cuts: list[Cut] = []
    rounds = 0
    violated = False
    while True:
        remaining = max(settings.time_limit - (time.perf_counter() - t0), 1e-3)
        raw = solve(model.with_cuts(cuts), settings.model_copy(update={"time_limit": remaining}))
        if raw.x is None or raw.status is not SolveStatus.OPTIMAL:
            violated = False
            break
        new = list(oracle(raw.x))
        violated = bool(new)
        if not new or rounds >= max_rounds or time.perf_counter() - t0 >= settings.time_limit:
            break
        cuts.extend(new)
        rounds += 1
    extra = {**raw.extra, "lazy_rounds": rounds, "lazy_cuts": len(cuts)}
    if violated:  # stopped early: the point is not feasible for the full model
        return replace(raw, status=SolveStatus.TIME_LIMIT, x=None, objective=None, extra=extra)
    duals = raw.duals[: model.num_constrs] if raw.duals is not None else None
    return replace(raw, duals=duals, extra=extra)


def create_backend(name: BackendName = BackendName.AUTO) -> SolverBackend:
    """Instantiate a backend; ``auto`` picks Gurobi if installed, else HiGHS."""
    resolved = resolve_backend(name)
    if resolved is BackendName.GUROBI:
        from linear_opt.solvers.gurobi import GurobiBackend

        return GurobiBackend()
    from linear_opt.solvers.highs import HighsBackend

    return HighsBackend()
