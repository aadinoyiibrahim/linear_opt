r"""Capacitated facility location (MILP).

Open facilities :math:`y_i \in \{0,1\}` (fixed cost :math:`f_i`, capacity
:math:`s_i`) and assign the fraction :math:`x_{ij} \in [0,1]` of customer
:math:`j`'s demand :math:`d_j` to facility :math:`i` at cost :math:`c_{ij}` (the
cost of serving *all* of :math:`j` from :math:`i`, as in OR-Library):

.. math::

    \min \sum_i f_i y_i + \sum_{i,j} c_{ij} x_{ij}
    \quad\text{s.t.}\quad
    \sum_i x_{ij} = 1, \qquad
    \sum_j d_j x_{ij} \le s_i y_i, \qquad
    \underbrace{x_{ij} \le y_i}_{\text{strong only}}, \qquad
    \underbrace{\textstyle\sum_i s_i y_i \ge \sum_j d_j}_{\text{optional cut}}.

**Strong vs weak.** Both formulations have the same integer solutions, but the
disaggregated linking :math:`x_{ij} \le y_i` cuts off fractional points where a
facility is "open by 3 %" and still serves a customer fully. The LP relaxation
of the strong model is therefore never weaker, typically much tighter, at the
price of :math:`mn` extra rows. ``linopt variants`` measures the difference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel, ConfigDict

from linear_opt.core.config import ProblemKind
from linear_opt.core.ir import (
    ConstrBlock,
    FloatArray,
    ModelBuilder,
    Sense,
    VarBlock,
    VarType,
    sum_over,
)
from linear_opt.core.model import OptimizationModel
from linear_opt.core.result import ProgressPoint, RawSolution
from linear_opt.data.distances import haversine_matrix
from linear_opt.data.geonames import City
from linear_opt.data.orlib import CapData

TOL = 1e-6


@dataclass(frozen=True, eq=False)
class FacilityInstance:
    """Immutable capacitated facility location instance.

    Attributes:
        facilities: Candidate names (length ``m``).
        customers: Customer names (length ``n``).
        capacity: Capacities ``s``, shape ``(m,)``.
        fixed_cost: Opening costs ``f``, shape ``(m,)``.
        demand: Demands ``d``, shape ``(n,)``.
        assign_cost: ``c[i, j]`` = cost of serving all of customer ``j`` from ``i``.
        facility_xy: Coordinates ``(m, 2)`` or ``None`` (OR-Library has none).
        customer_xy: Coordinates ``(n, 2)`` or ``None``.
        geographic: Whether coordinates are ``[lat, lon]``.
        reference: Published optimal objective, if known.
    """

    facilities: tuple[str, ...]
    customers: tuple[str, ...]
    capacity: FloatArray
    fixed_cost: FloatArray
    demand: FloatArray
    assign_cost: FloatArray
    facility_xy: FloatArray | None = None
    customer_xy: FloatArray | None = None
    geographic: bool = False
    reference: float | None = None

    def __post_init__(self) -> None:
        m, n = len(self.facilities), len(self.customers)
        expected = {
            "capacity": (self.capacity.shape, (m,)),
            "fixed_cost": (self.fixed_cost.shape, (m,)),
            "demand": (self.demand.shape, (n,)),
            "assign_cost": (self.assign_cost.shape, (m, n)),
        }
        if self.facility_xy is not None:
            expected["facility_xy"] = (self.facility_xy.shape, (m, 2))
        if self.customer_xy is not None:
            expected["customer_xy"] = (self.customer_xy.shape, (n, 2))
        for label, (got, want) in expected.items():
            if got != want:
                raise ValueError(f"{label} has shape {got}, expected {want}")
        arrays = (self.capacity, self.fixed_cost, self.demand, self.assign_cost)
        if any(np.any(a < 0) or not np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("capacities, costs and demands must be finite and non-negative")

    @property
    def shape(self) -> tuple[int, int]:
        """``(m, n)``."""
        return len(self.facilities), len(self.customers)

    @property
    def total_demand(self) -> float:
        """Sum of demands."""
        return float(self.demand.sum())

    # ----------------------------------------------------------------- factories
    @classmethod
    def from_orlib(cls, data: CapData, reference: float | None = None) -> FacilityInstance:
        """Wrap an OR-Library ``cap*`` instance."""
        m, n = data.shape
        return cls(
            tuple(f"W{i + 1}" for i in range(m)),
            tuple(f"C{j + 1}" for j in range(n)),
            data.capacity,
            data.fixed_cost,
            data.demand,
            data.cost,
            reference=reference,
        )

    @classmethod
    def random(
        cls, n_facilities: int, n_customers: int, *, seed: int = 0, capacity_ratio: float = 3.0
    ) -> FacilityInstance:
        """Random Euclidean instance; total capacity = ``capacity_ratio`` x demand."""
        rng = np.random.default_rng(seed)
        fac, cus = rng.random((n_facilities, 2)), rng.random((n_customers, 2))
        demand = rng.integers(5, 35, n_customers).astype(np.float64)
        capacity = np.full(n_facilities, capacity_ratio * demand.sum() / n_facilities)
        fixed = rng.uniform(50.0, 150.0, n_facilities)
        dist = np.linalg.norm(fac[:, None, :] - cus[None, :, :], axis=2)
        return cls(
            tuple(f"F{i}" for i in range(n_facilities)),
            tuple(f"C{j}" for j in range(n_customers)),
            capacity,
            fixed,
            demand,
            10.0 * dist * demand[None, :],
            fac,
            cus,
        )

    @classmethod
    def from_cities(
        cls,
        cities: Sequence[City],
        *,
        n_candidates: int,
        capacity_ratio: float = 2.5,
        fixed_cost: float = 50_000.0,
        unit_cost: float = 1.0,
    ) -> FacilityInstance:
        """Real geography, stylised economics.

        Customers are all ``cities`` (demand = population / 1000); candidate sites
        are the ``n_candidates`` largest. Site capacity is proportional to its
        population and sums to ``capacity_ratio * total_demand``; opening cost is
        ``fixed_cost * sqrt(pop_i / median pop)`` (larger cities, dearer land);
        serving cost is ``unit_cost * demand_j * distance_ij`` (km).
        """
        if not 0 < n_candidates <= len(cities):
            raise ValueError("need 0 < n_candidates <= number of cities")
        ranked = sorted(cities, key=lambda c: -c.population)
        sites = ranked[:n_candidates]
        demand = np.array([c.population for c in ranked], dtype=np.float64) / 1000.0
        site_pop = np.array([c.population for c in sites], dtype=np.float64)
        site_xy = np.array([[c.lat, c.lon] for c in sites])
        cust_xy = np.array([[c.lat, c.lon] for c in ranked])
        dist = haversine_matrix(site_xy, cust_xy)
        return cls(
            tuple(c.name for c in sites),
            tuple(c.name for c in ranked),
            capacity_ratio * demand.sum() * site_pop / site_pop.sum(),
            fixed_cost * np.sqrt(site_pop / np.median(site_pop)),
            demand,
            unit_cost * dist * demand[None, :],
            site_xy,
            cust_xy,
            geographic=True,
        )


class FacilityOptions(BaseModel):
    """Options in the ``[model]`` table of a facility-location config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    formulation: Literal["strong", "weak"] = "strong"
    aggregate_capacity_cut: bool = True
    single_sourcing: bool = False
    warm_start: Literal["greedy", "none"] = "greedy"


@dataclass(frozen=True, eq=False)
class FacilitySolution:
    """Open facilities, assignment fractions and cost split."""

    open: np.ndarray[Any, np.dtype[np.bool_]]
    assignment: FloatArray  # (m, n) fractions
    fixed_part: float
    assignment_part: float
    objective: float
    progress: tuple[ProgressPoint, ...] = ()

    @property
    def total_cost(self) -> float:
        """Fixed plus assignment cost (recomputed from the parts)."""
        return self.fixed_part + self.assignment_part

    def load(self, demand: FloatArray) -> FloatArray:
        """Demand served by each facility."""
        return np.asarray(self.assignment @ demand, dtype=np.float64)

    def main_server(self) -> np.ndarray[Any, np.dtype[np.int64]]:
        """Facility serving the largest share of each customer."""
        return np.asarray(self.assignment.argmax(axis=0), dtype=np.int64)


def greedy_solution(
    inst: FacilityInstance, single_sourcing: bool = False
) -> tuple[np.ndarray[Any, np.dtype[np.bool_]], FloatArray] | None:
    """Construct a feasible solution: open by unit cost, assign cheapest-first.

    Facilities are ranked by ``f_i / s_i + mean_j c_ij / d_j`` (cost per unit of
    demand) and opened until capacity covers demand; customers, largest first,
    then fill their cheapest open facilities. If single-sourcing leaves someone
    unserved, the next facility is opened and the assignment restarts.

    Returns:
        ``(open, assignment)`` or ``None`` if no feasible greedy solution exists.
    """
    m, n = inst.shape
    unit = inst.assign_cost / np.maximum(inst.demand, TOL)[None, :]
    rank = np.argsort(inst.fixed_cost / np.maximum(inst.capacity, TOL) + unit.mean(axis=1))
    k = int(np.searchsorted(np.cumsum(inst.capacity[rank]), inst.total_demand - TOL)) + 1
    while k <= m:
        is_open = np.zeros(m, dtype=bool)
        is_open[rank[:k]] = True
        left = np.where(is_open, inst.capacity, 0.0)
        x = np.zeros((m, n))
        ok = True
        for j in np.argsort(-inst.demand):
            need = inst.demand[j]
            for i in np.argsort(unit[:, j]):
                if not is_open[i] or left[i] <= TOL:
                    continue
                if single_sourcing:
                    if left[i] >= need - TOL:
                        x[i, j], left[i], need = 1.0, left[i] - need, 0.0
                        break
                    continue
                take = min(need, left[i])
                x[i, j] += take / inst.demand[j] if inst.demand[j] > 0 else 0.0
                left[i] -= take
                need -= take
                if need <= TOL:
                    break
            if need > TOL:
                ok = False
                break
            if inst.demand[j] == 0 and x[:, j].sum() == 0:
                x[int(np.flatnonzero(is_open)[0]), j] = 1.0
        if ok:
            return is_open, x
        k += 1
    return None


class FacilityModel(OptimizationModel[FacilityInstance, FacilitySolution, FacilityOptions]):
    """Capacitated facility location on a :class:`FacilityInstance`."""

    kind: ClassVar[ProblemKind] = ProblemKind.FACILITY_LOCATION
    options_type: ClassVar[type[BaseModel]] = FacilityOptions

    _open: VarBlock
    _assign: VarBlock
    _demand_rows: ConstrBlock

    def _build(self, builder: ModelBuilder) -> None:
        inst, opts = self.data, self.options
        m, n = inst.shape
        self._open = builder.add_vars("open", m, vtype=VarType.BINARY, obj=inst.fixed_cost)
        self._assign = builder.add_vars(
            "assign",
            (m, n),
            ub=1.0,
            vtype=VarType.BINARY if opts.single_sourcing else VarType.CONTINUOUS,
            obj=inst.assign_cost,
        )
        x, y = self._assign, self._open
        self._demand_rows = builder.add_constrs(
            "serve", [(x, sum_over((m, n), axis=0))], Sense.EQ, 1.0
        )
        builder.add_constrs(
            "capacity",
            [
                (x, sp.kron(sp.eye_array(m), inst.demand[None, :])),
                (y, -sp.diags_array(inst.capacity)),
            ],
            Sense.LE,
            0.0,
        )
        if opts.formulation == "strong":
            builder.add_constrs(
                "link",
                [(x, sp.eye_array(m * n)), (y, -sp.kron(sp.eye_array(m), np.ones((n, 1))))],
                Sense.LE,
                0.0,
            )
        if opts.aggregate_capacity_cut:
            builder.add_constrs("cover", [(y, inst.capacity[None, :])], Sense.GE, inst.total_demand)

    def warm_start(self) -> FloatArray | None:
        """Greedy start (MIP start) unless disabled."""
        if self.options.warm_start == "none":
            return None
        found = greedy_solution(self.data, self.options.single_sourcing)
        if found is None:
            return None
        is_open, x = found
        start = np.full(self.build().num_vars, np.nan)
        start[self._open.slice] = is_open.astype(np.float64)
        start[self._assign.slice] = x.ravel()
        return start

    def reference_objective(self) -> float | None:
        """Published optimum; only valid for the splittable problem it was computed for."""
        return None if self.options.single_sourcing else self.data.reference

    @classmethod
    def variants(cls) -> dict[str, dict[str, Any]]:
        """Formulations compared by ``linopt variants``."""
        return {
            "strong": {"formulation": "strong", "aggregate_capacity_cut": True},
            "weak + cut": {"formulation": "weak", "aggregate_capacity_cut": True},
            "weak": {"formulation": "weak", "aggregate_capacity_cut": False},
        }

    def _extract(self, raw: RawSolution) -> FacilitySolution:
        assert raw.x is not None
        assert raw.objective is not None
        is_open = self._open.values(raw.x) > 0.5
        x = np.clip(self._assign.values(raw.x), 0.0, 1.0)
        x[~is_open, :] = np.where(x[~is_open, :] < 1e-7, 0.0, x[~is_open, :])
        return FacilitySolution(
            open=is_open,
            assignment=x,
            fixed_part=float(self.data.fixed_cost[is_open].sum()),
            assignment_part=float((self.data.assign_cost * x).sum()),
            objective=raw.objective,
            progress=raw.progress,
        )

    def validate(self, solution: FacilitySolution) -> list[str]:
        """Re-check every constraint and the objective from the instance data."""
        inst, x = self.data, solution.assignment
        names_f, names_c = inst.facilities, inst.customers
        issues: list[str] = []
        served = x.sum(axis=0)
        for j in np.flatnonzero(np.abs(served - 1.0) > 1e-5):
            issues.append(f"customer {names_c[j]} is served {served[j]:.4f} times, expected 1")
        closed_use = x[~solution.open].sum(axis=1)
        for k, i in enumerate(np.flatnonzero(~solution.open)):
            if closed_use[k] > 1e-5:
                issues.append(f"closed facility {names_f[i]} serves customers")
        load = solution.load(inst.demand)
        for i in np.flatnonzero(load > inst.capacity * (1 + 1e-6) + 1e-6):
            issues.append(
                f"facility {names_f[i]} load {load[i]:.4g} > capacity {inst.capacity[i]:.4g}"
            )
        if self.options.single_sourcing:
            fractional = np.abs(x - np.round(x)) > 1e-5
            if fractional.any():
                issues.append(
                    f"{int(fractional.sum())} fractional assignments under single sourcing"
                )
        if not np.isclose(solution.total_cost, solution.objective, rtol=1e-6, atol=1e-4):
            issues.append(
                f"objective {solution.objective:.6f} != recomputed cost {solution.total_cost:.6f}"
            )
        return issues
