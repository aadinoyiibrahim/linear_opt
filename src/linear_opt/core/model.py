"""Base class for every optimisation model: build -> solve -> extract -> validate."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel

from linear_opt.core.config import BackendName, ProblemKind, SolverSettings
from linear_opt.core.ir import FloatArray, LinearModel, ModelBuilder, ObjSense
from linear_opt.core.result import LazyOracle, RawSolution

D = TypeVar("D")  # instance data
S = TypeVar("S")  # typed solution
O = TypeVar("O", bound=BaseModel)  # noqa: E741 - model options


@dataclass(frozen=True, eq=False)
class Result(Generic[S]):
    """Outcome of :meth:`OptimizationModel.solve`."""

    name: str
    kind: ProblemKind
    raw: RawSolution
    solution: S | None
    violations: tuple[str, ...]
    model_stats: Mapping[str, Any] = field(default_factory=dict)
    reference: float | None = None  # known optimum (e.g. from a benchmark library)

    @property
    def ok(self) -> bool:
        """A solution exists and passed independent validation."""
        return self.solution is not None and not self.violations

    @property
    def reference_gap(self) -> float | None:
        """Relative deviation of the objective from the known optimum."""
        if self.reference is None or self.raw.objective is None:
            return None
        return (self.raw.objective - self.reference) / max(abs(self.reference), 1e-10)

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable overview of the run."""
        raw = self.raw
        return {
            "name": self.name,
            "kind": self.kind.value,
            "status": raw.status.value,
            "objective": raw.objective,
            "bound": raw.bound,
            "gap": raw.gap,
            "backend": f"{raw.backend} {raw.backend_version}",
            "runtime_s": round(raw.runtime, 4),
            "iterations": raw.iterations,
            "nodes": raw.nodes,
            "model": dict(self.model_stats),
            "violations": list(self.violations),
            "reference": self.reference,
            "reference_gap": self.reference_gap,
            **{k: v for k, v in raw.extra.items() if isinstance(v, int | float | str)},
        }


class OptimizationModel(ABC, Generic[D, S, O]):
    """Template for a problem class.

    Subclasses implement three things:

    * :meth:`_build` - declare variables/constraints on a :class:`ModelBuilder`;
    * :meth:`_extract` - turn the raw vector into a typed, domain-level solution;
    * :meth:`validate` - re-check feasibility *from the instance data*, so a
      modelling bug cannot hide behind a solver reporting "optimal".

    Optional hooks: :meth:`lazy_constraints` and :meth:`warm_start`.
    """

    kind: ClassVar[ProblemKind]
    options_type: ClassVar[type[BaseModel]]
    sense: ClassVar[ObjSense] = ObjSense.MINIMIZE

    def __init__(
        self, data: D, options: O | Mapping[str, Any] | None = None, *, name: str | None = None
    ) -> None:
        self.data = data
        self.options: O = (
            options
            if isinstance(options, BaseModel)
            else self.options_type.model_validate(dict(options or {}))  # type: ignore[assignment]
        )
        self.name = name or self.kind.value
        self._linear_model: LinearModel | None = None

    # ---------------------------------------------------------------- template
    @abstractmethod
    def _build(self, builder: ModelBuilder) -> None:
        """Declare variable and constraint blocks."""

    @abstractmethod
    def _extract(self, raw: RawSolution) -> S:
        """Convert a raw solution (with ``raw.x`` set) into the typed solution."""

    def validate(self, solution: S) -> list[str]:
        """Return human-readable constraint violations (empty if feasible)."""
        return []

    def lazy_constraints(self) -> LazyOracle | None:
        """Separation oracle for lazy constraints, if the model uses them."""
        return None

    def warm_start(self) -> FloatArray | None:
        """Heuristic starting point for MIPs."""
        return None

    def user_cuts(self) -> LazyOracle | None:
        """Separator for fractional points (cutting planes), if the model has one."""
        return None

    def reference_objective(self) -> float | None:
        """Known optimal objective of this instance, if published."""
        return None

    @classmethod
    def variants(cls) -> dict[str, dict[str, Any]]:
        """Named option sets worth comparing (e.g. alternative formulations)."""
        return {}

    def with_options(self, **overrides: Any) -> OptimizationModel[D, S, O]:
        """A fresh model on the same data with some options changed."""
        options = {**self.options.model_dump(), **overrides}
        return type(self)(self.data, options, name=self.name)

    # -------------------------------------------------------------------- API
    def build(self) -> LinearModel:
        """Compile (once) to the solver-neutral IR."""
        if self._linear_model is None:
            builder = ModelBuilder(self.name, self.sense)
            self._build(builder)
            self._linear_model = builder.build()
        return self._linear_model

    def solve(
        self,
        settings: SolverSettings | None = None,
        backend: Any = None,
    ) -> Result[S]:
        """Build, solve, extract and validate.

        Args:
            settings: Solver settings (defaults if omitted).
            backend: A :class:`~linear_opt.solvers.SolverBackend` instance, a
                :class:`BackendName`, or ``None`` to use ``settings.backend``.
        """
        from linear_opt.solvers import SolverBackend, create_backend

        settings = settings or SolverSettings()
        if not isinstance(backend, SolverBackend):
            backend = create_backend(BackendName(backend or settings.backend))
        lm = self.build()
        raw = backend.solve(
            lm,
            settings,
            lazy=self.lazy_constraints(),
            warm_start=self.warm_start(),
            user_cuts=self.user_cuts(),
        )
        solution = self._extract(raw) if raw.has_solution else None
        violations = tuple(self.validate(solution)) if solution is not None else ()
        return Result(
            self.name, self.kind, raw, solution, violations, lm.stats(), self.reference_objective()
        )
