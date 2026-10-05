r"""Transportation problem / discrete optimal transport (LP).

Given depots :math:`i` with capacity :math:`s_i`, customers :math:`j` with
demand :math:`d_j` and unit costs :math:`c_{ij}`:

.. math::

    \min_{x \ge 0} \sum_{i,j} c_{ij} x_{ij}
    \quad\text{s.t.}\quad
    \sum_j x_{ij} \le s_i \;\; (\text{supply}, \text{dual } u_i \le 0), \qquad
    \sum_i x_{ij} \ge d_j \;\; (\text{demand}, \text{dual } v_j \ge 0).

With :math:`\sum s = \sum d` and equality constraints this is exactly the
Kantorovich formulation of discrete optimal transport between the two mass
distributions.

**Reading the duals.** :math:`v_j` is the marginal cost of delivering one more
unit to customer :math:`j`; :math:`-u_i` is what one more unit of capacity at
depot :math:`i` would save. Complementary slackness gives the economic
interpretation: a route is used only if :math:`c_{ij} = u_i + v_j`, i.e. its
reduced cost is zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from linear_opt.core.config import ProblemKind
from linear_opt.core.ir import ConstrBlock, FloatArray, ModelBuilder, Sense, VarBlock, sum_over
from linear_opt.core.model import OptimizationModel
from linear_opt.core.result import RawSolution
from linear_opt.data.distances import haversine_matrix, road_distance_matrix
from linear_opt.data.geonames import City

TOL = 1e-6


@dataclass(frozen=True, eq=False)
class TransportInstance:
    """Immutable transport instance.

    Attributes:
        sources: Depot names (length ``m``).
        sinks: Customer names (length ``n``).
        supply: Capacities, shape ``(m,)``.
        demand: Demands, shape ``(n,)``.
        cost: Unit costs, shape ``(m, n)``.
        source_xy: Coordinates of depots, ``(m, 2)``; ``[lat, lon]`` if ``geographic``.
        sink_xy: Coordinates of customers, ``(n, 2)``.
        geographic: Whether coordinates are latitude/longitude.
        cost_unit: Label for the cost unit (e.g. ``"km"``).
    """

    sources: tuple[str, ...]
    sinks: tuple[str, ...]
    supply: FloatArray
    demand: FloatArray
    cost: FloatArray
    source_xy: FloatArray
    sink_xy: FloatArray
    geographic: bool = False
    cost_unit: str = "distance"

    def __post_init__(self) -> None:
        m, n = len(self.sources), len(self.sinks)
        checks = {
            "supply": (self.supply.shape, (m,)),
            "demand": (self.demand.shape, (n,)),
            "cost": (self.cost.shape, (m, n)),
            "source_xy": (self.source_xy.shape, (m, 2)),
            "sink_xy": (self.sink_xy.shape, (n, 2)),
        }
        for label, (got, want) in checks.items():
            if got != want:
                raise ValueError(f"{label} has shape {got}, expected {want}")
        if min(self.supply.min(), self.demand.min()) < 0:
            raise ValueError("supply and demand must be non-negative")
        if not np.all(np.isfinite(self.cost)):
            raise ValueError("costs must be finite")

    @property
    def shape(self) -> tuple[int, int]:
        """``(m, n)``."""
        return len(self.sources), len(self.sinks)

    @property
    def is_balanced(self) -> bool:
        """Total supply equals total demand."""
        return bool(np.isclose(self.supply.sum(), self.demand.sum()))

    # ----------------------------------------------------------------- factories
    @classmethod
    def random(
        cls, n_sources: int, n_sinks: int, *, seed: int = 0, slack: float = 0.1
    ) -> TransportInstance:
        """Uniform points in the unit square, Euclidean costs, integer demands."""
        rng = np.random.default_rng(seed)
        src, snk = rng.random((n_sources, 2)), rng.random((n_sinks, 2))
        demand = rng.integers(10, 100, n_sinks).astype(np.float64)
        weights = rng.random(n_sources) + 0.5
        supply = (1 + slack) * demand.sum() * weights / weights.sum()
        cost = np.linalg.norm(src[:, None, :] - snk[None, :, :], axis=2)
        return cls(
            tuple(f"S{i}" for i in range(n_sources)),
            tuple(f"D{j}" for j in range(n_sinks)),
            supply,
            demand,
            cost,
            src,
            snk,
            geographic=False,
            cost_unit="distance",
        )

    @classmethod
    def from_cities(
        cls,
        cities: Sequence[City],
        *,
        n_sources: int,
        seed: int = 0,
        slack: float = 0.1,
        cost: Literal["haversine", "road"] = "haversine",
    ) -> TransportInstance:
        """Split cities into depots and customers; mass is population (thousands).

        ``n_sources`` depots are drawn at random (reproducibly via ``seed``); the
        remaining cities are customers with demand = population / 1000. Total
        depot capacity is ``(1 + slack)`` x total demand, split proportionally to
        depot population.
        """
        if not 0 < n_sources < len(cities):
            raise ValueError("need 0 < n_sources < number of cities")
        if slack < 0:
            raise ValueError("slack must be >= 0 (otherwise the instance is infeasible)")
        rng = np.random.default_rng(seed)
        is_source = np.zeros(len(cities), dtype=bool)
        is_source[rng.choice(len(cities), n_sources, replace=False)] = True
        src = [c for c, s in zip(cities, is_source, strict=True) if s]
        snk = [c for c, s in zip(cities, is_source, strict=True) if not s]

        demand = np.array([c.population for c in snk], dtype=np.float64) / 1000.0
        weight = np.array([c.population for c in src], dtype=np.float64)
        supply = (1 + slack) * demand.sum() * weight / weight.sum()
        src_xy = np.array([[c.lat, c.lon] for c in src])
        snk_xy = np.array([[c.lat, c.lon] for c in snk])
        dist = (
            haversine_matrix(src_xy, snk_xy)
            if cost == "haversine"
            else road_distance_matrix(src_xy, snk_xy)
        )
        return cls(
            tuple(c.name for c in src),
            tuple(c.name for c in snk),
            supply,
            demand,
            dist,
            src_xy,
            snk_xy,
            geographic=True,
            cost_unit="km" if cost == "haversine" else "road km",
        )


class TransportOptions(BaseModel):
    """Options in the ``[model]`` table of a transport config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    supply_constraint: Literal["le", "eq"] = "le"


@dataclass(frozen=True, eq=False)
class TransportSolution:
    """Optimal flows plus the dual (price) information of the LP."""

    flow: FloatArray
    total_cost: float
    supply_duals: FloatArray | None
    demand_duals: FloatArray | None
    reduced_costs: FloatArray | None

    def routes(self, tol: float = TOL) -> list[tuple[int, int, float]]:
        """Used routes ``(i, j, flow)``, largest flow first."""
        i, j = np.nonzero(self.flow > tol)
        order = np.argsort(-self.flow[i, j])
        return [(int(i[k]), int(j[k]), float(self.flow[i[k], j[k]])) for k in order]

    def utilisation(self, supply: FloatArray) -> FloatArray:
        """Fraction of each depot's capacity in use."""
        return np.divide(self.flow.sum(axis=1), supply, out=np.zeros_like(supply), where=supply > 0)


class TransportModel(OptimizationModel[TransportInstance, TransportSolution, TransportOptions]):
    """Transportation LP on a :class:`TransportInstance`."""

    kind: ClassVar[ProblemKind] = ProblemKind.TRANSPORT
    options_type: ClassVar[type[BaseModel]] = TransportOptions

    _flow: VarBlock
    _supply: ConstrBlock
    _demand: ConstrBlock

    def _build(self, builder: ModelBuilder) -> None:
        inst = self.data
        shape = inst.shape
        self._flow = builder.add_vars("flow", shape, obj=inst.cost)
        supply_sense = Sense.EQ if self.options.supply_constraint == "eq" else Sense.LE
        self._supply = builder.add_constrs(
            "supply", [(self._flow, sum_over(shape, axis=1))], supply_sense, inst.supply
        )
        self._demand = builder.add_constrs(
            "demand", [(self._flow, sum_over(shape, axis=0))], Sense.GE, inst.demand
        )

    def _extract(self, raw: RawSolution) -> TransportSolution:
        assert raw.x is not None
        assert raw.objective is not None
        y, z = raw.duals, raw.reduced_costs
        return TransportSolution(
            flow=np.maximum(self._flow.values(raw.x), 0.0),
            total_cost=raw.objective,
            supply_duals=self._supply.values(y) if y is not None else None,
            demand_duals=self._demand.values(y) if y is not None else None,
            reduced_costs=self._flow.values(z) if z is not None else None,
        )

    def validate(self, solution: TransportSolution) -> list[str]:
        """Check capacities, demands and the objective against the raw data."""
        inst, flow = self.data, solution.flow
        scale = max(1.0, float(inst.demand.max(initial=0.0)))
        tol = TOL * scale
        issues: list[str] = []
        shipped, received = flow.sum(axis=1), flow.sum(axis=0)
        for i in np.nonzero(shipped > inst.supply + tol)[0]:
            issues.append(
                f"depot {inst.sources[i]} ships {shipped[i]:.4g} > capacity {inst.supply[i]:.4g}"
            )
        for j in np.nonzero(received < inst.demand - tol)[0]:
            issues.append(
                f"customer {inst.sinks[j]} gets {received[j]:.4g} < demand {inst.demand[j]:.4g}"
            )
        recomputed = float((inst.cost * flow).sum())
        if not np.isclose(recomputed, solution.total_cost, rtol=1e-6, atol=1e-6):
            issues.append(
                f"objective {solution.total_cost:.6g} != recomputed cost {recomputed:.6g}"
            )
        return issues
