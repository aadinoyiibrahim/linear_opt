r"""Formulation studies: compare model variants on the same instance.

For a minimisation MIP with optimum :math:`z^*` and LP-relaxation value
:math:`z_{LP}`, the **root gap** :math:`(z^* - z_{LP}) / |z^*|` measures how tight
a formulation is. Tighter formulations usually mean smaller search trees, which
is the classic argument for "strong" over "weak" (aggregated) constraints.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from linear_opt.core.config import SolverSettings
from linear_opt.core.model import OptimizationModel, Result
from linear_opt.core.result import RawSolution


@dataclass(frozen=True, eq=False)
class VariantResult:
    """LP relaxation and MIP result of one model variant."""

    name: str
    options: Mapping[str, Any]
    relaxation: RawSolution
    result: Result[Any]

    @property
    def lp_bound(self) -> float | None:
        """Objective of the LP relaxation."""
        return self.relaxation.objective

    @property
    def root_gap(self) -> float | None:
        """Relative gap between the LP relaxation and the best MIP objective."""
        best, lp = self.result.raw.objective, self.lp_bound
        if best is None or lp is None:
            return None
        return abs(best - lp) / max(abs(best), 1e-10)

    def row(self) -> dict[str, Any]:
        """Flat record for tables and JSON."""
        raw = self.result.raw
        return {
            "variant": self.name,
            "lp_bound": self.lp_bound,
            "root_gap": self.root_gap,
            "objective": raw.objective,
            "status": raw.status.value,
            "runtime_s": raw.runtime,
            "nodes": raw.nodes,
            "variables": self.result.model_stats.get("variables"),
            "constraints": self.result.model_stats.get("constraints"),
            "valid": self.result.ok,
        }


def run_variants(
    model: OptimizationModel[Any, Any, Any],
    settings: SolverSettings,
    backend: Any = None,
    variants: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[VariantResult]:
    """Solve the LP relaxation and the MIP for every variant of ``model``.

    Args:
        model: Any model; its data are reused, options are overridden per variant.
        settings: Solver settings applied to every solve.
        backend: Backend instance or name (defaults to ``settings.backend``).
        variants: Option overrides by name; defaults to ``type(model).variants()``.
    """
    from linear_opt.core.config import BackendName
    from linear_opt.solvers import SolverBackend, create_backend

    solver = (
        backend
        if isinstance(backend, SolverBackend)
        else create_backend(BackendName(backend or settings.backend))
    )
    out: list[VariantResult] = []
    for name, overrides in (variants or type(model).variants()).items():
        variant = model.with_options(**overrides)
        # The oracle (if any) is applied by row generation, so a formulation with
        # lazy constraints is judged by the LP bound of its *full* constraint set.
        relaxation = solver.solve(
            variant.build().relaxed(), settings, lazy=variant.lazy_constraints()
        )
        out.append(
            VariantResult(name, dict(overrides), relaxation, variant.solve(settings, solver))
        )
    return out
