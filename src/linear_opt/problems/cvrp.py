r"""Capacitated vehicle routing problem (MILP).

Customers :math:`1..n-1` with demands :math:`d_i` are served from depot
:math:`0` by at most :math:`K` vehicles of capacity :math:`Q`.

* **Two-index** (Laporte-Nobert): integer :math:`x_e` per edge, :math:`x_{0j}
  \in \{0,1,2\}` (2 = out-and-back route), degree 2 at customers, and the
  *rounded capacity inequalities*
  :math:`x(\delta(S)) \ge 2 \lceil d(S) / Q \rceil` added lazily. On integer
  points, the customer sets of the connected components (after removing the
  depot) are exactly the routes and subtours, so this separation is exact; on
  fractional points it is a standard heuristic.
* **MTZ load**: directed arcs and load variables
  :math:`u_i - u_j + Q x_{ij} \le Q - d_j` - compact but weak.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, ClassVar, Literal

import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel, ConfigDict

from linear_opt.core.config import ProblemKind
from linear_opt.core.graph import EdgeIndex, components, mask
from linear_opt.core.ir import FloatArray, ModelBuilder, Sense, VarBlock, VarType, sum_over
from linear_opt.core.model import OptimizationModel
from linear_opt.core.result import Cut, LazyOracle, ProgressPoint, RawSolution
from linear_opt.data.distances import haversine_matrix
from linear_opt.data.geonames import City
from linear_opt.data.tsplib import TsplibProblem
from linear_opt.problems.heuristics import clarke_wright

EPS = 1e-6


@dataclass(frozen=True, eq=False)
class CVRPInstance:
    """Immutable CVRP instance; node 0 is the depot."""

    names: tuple[str, ...]
    dist: FloatArray
    demand: FloatArray
    capacity: float
    vehicles: int
    coords: FloatArray | None = None
    geographic: bool = False
    reference: float | None = None
    reference_routes: tuple[tuple[int, ...], ...] | None = None
    unit: str = "distance"

    def __post_init__(self) -> None:
        n = len(self.names)
        if n < 2:
            raise ValueError("need a depot and at least one customer")
        if self.dist.shape != (n, n) or self.demand.shape != (n,):
            raise ValueError("dist must be (n, n) and demand (n,)")
        if not np.allclose(self.dist, self.dist.T):
            raise ValueError("distance matrix must be symmetric")
        if self.demand[0] != 0:
            raise ValueError("depot demand must be 0")
        if np.any(self.demand < 0) or self.demand.max() > self.capacity:
            raise ValueError("every demand must be in [0, capacity]")
        if self.vehicles < self.min_vehicles:
            raise ValueError(
                f"{self.vehicles} vehicles cannot carry total demand {self.demand.sum():g} "
                f"(need >= {self.min_vehicles})"
            )

    @property
    def n(self) -> int:
        """Nodes including the depot."""
        return len(self.names)

    @property
    def min_vehicles(self) -> int:
        """Lower bound ``ceil(total demand / Q)`` on the number of routes."""
        return max(1, math.ceil(float(self.demand.sum()) / self.capacity - 1e-9))

    def route_cost(self, route: Sequence[int]) -> float:
        """Cost of depot -> route -> depot."""
        path = [0, *route, 0]
        return float(sum(self.dist[a, b] for a, b in pairwise(path)))

    # ----------------------------------------------------------------- factories
    @classmethod
    def from_tsplib(
        cls,
        problem: TsplibProblem,
        reference: float | None = None,
        routes: Sequence[Sequence[int]] | None = None,
    ) -> CVRPInstance:
        """Wrap a CVRPLIB problem; nodes are permuted so the depot comes first."""
        if problem.demand is None or problem.capacity is None or problem.depot is None:
            raise ValueError(f"{problem.name} is not a CVRP instance")
        n, depot = problem.dimension, problem.depot
        order = np.r_[depot, [v for v in range(n) if v != depot]]
        if depot != 0 and routes is not None:
            raise ValueError("reference routes assume the depot is node 1")
        vehicles = problem.vehicles or math.ceil(problem.demand.sum() / problem.capacity)
        return cls(
            tuple(str(v + 1) for v in order),
            problem.distances[np.ix_(order, order)],
            problem.demand[order],
            problem.capacity,
            vehicles,
            problem.coords[order] if problem.coords is not None else None,
            reference=reference,
            reference_routes=tuple(tuple(r) for r in routes) if routes else None,
        )

    @classmethod
    def random(
        cls, n_customers: int, *, seed: int = 0, capacity: float = 100.0, spare_vehicles: int = 1
    ) -> CVRPInstance:
        """Depot at the centre of a 100 x 100 square; integer demands 1..30."""
        rng = np.random.default_rng(seed)
        pts = np.vstack([[50.0, 50.0], rng.random((n_customers, 2)) * 100])
        demand = np.r_[0.0, rng.integers(1, 31, n_customers).astype(np.float64)]
        dist = np.floor(np.linalg.norm(pts[:, None] - pts[None, :], axis=2) + 0.5)
        k = math.ceil(demand.sum() / capacity) + spare_vehicles
        names = ("depot", *(f"C{i}" for i in range(1, n_customers + 1)))
        return cls(names, dist, demand, capacity, k, pts)

    @classmethod
    def from_cities(
        cls,
        cities: Sequence[City],
        *,
        depot: str,
        capacity: float,
        vehicles: int,
        people_per_unit: float = 25_000.0,
    ) -> CVRPInstance:
        """Deliveries from one city to the others; demand = population / ``people_per_unit``."""
        names = [c.name for c in cities]
        if depot not in names:
            raise ValueError(f"depot {depot!r} is not among the selected cities")
        ordered = [cities[names.index(depot)], *(c for c in cities if c.name != depot)]
        xy = np.array([[c.lat, c.lon] for c in ordered])
        demand = np.r_[0.0, [math.ceil(c.population / people_per_unit) for c in ordered[1:]]]
        dist = np.floor(haversine_matrix(xy, xy) + 0.5)
        return cls(
            tuple(c.name for c in ordered), dist, demand, capacity, vehicles, xy,
            geographic=True, unit="km",
        )  # fmt: skip


class CVRPOptions(BaseModel):
    """Options in the ``[model]`` table of a CVRP config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    formulation: Literal["two_index", "mtz"] = "two_index"
    fleet: Literal["at_most", "exact"] = "at_most"
    warm_start: Literal["savings", "none"] = "savings"
    root_cuts: bool = True


@dataclass(frozen=True, eq=False)
class CVRPSolution:
    """Routes (customer sequences), their loads and the total cost."""

    routes: tuple[tuple[int, ...], ...]
    loads: tuple[float, ...]
    costs: tuple[float, ...]
    objective: float
    progress: tuple[ProgressPoint, ...] = ()

    @property
    def total_cost(self) -> float:
        """Sum of route costs (recomputed)."""
        return float(sum(self.costs))


class CVRPModel(OptimizationModel[CVRPInstance, CVRPSolution, CVRPOptions]):
    """CVRP on a :class:`CVRPInstance`."""

    kind: ClassVar[ProblemKind] = ProblemKind.CVRP
    options_type: ClassVar[type[BaseModel]] = CVRPOptions

    _x: VarBlock
    _load: VarBlock | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.edge_index = EdgeIndex(self.data.n)

    # ------------------------------------------------------------------ build
    def _depot_bounds(self) -> tuple[float, float]:
        k_max = float(self.data.vehicles)
        k_min = k_max if self.options.fleet == "exact" else float(self.data.min_vehicles)
        return k_min, k_max

    def _build(self, builder: ModelBuilder) -> None:
        inst, n = self.data, self.data.n
        k_min, k_max = self._depot_bounds()
        self._load = None
        if self.options.formulation == "two_index":
            e = self.edge_index
            ub = np.where(e.i == 0, 2.0, 1.0)
            self._x = builder.add_vars(
                "edge", e.m, ub=ub, vtype=VarType.INTEGER, obj=e.weights(inst.dist)
            )
            inc = e.incidence()
            builder.add_constrs("degree", [(self._x, inc[1:])], Sense.EQ, 2.0)
            builder.add_constrs("fleet_max", [(self._x, inc[:1])], Sense.LE, 2 * k_max)
            builder.add_constrs("fleet_min", [(self._x, inc[:1])], Sense.GE, 2 * k_min)
            return

        ub = 1.0 - np.eye(n)
        self._x = builder.add_vars("arc", (n, n), ub=ub, vtype=VarType.BINARY, obj=inst.dist)
        out, into = sum_over((n, n), axis=1), sum_over((n, n), axis=0)
        builder.add_constrs("leave", [(self._x, out[1:])], Sense.EQ, 1.0)
        builder.add_constrs("enter", [(self._x, into[1:])], Sense.EQ, 1.0)
        builder.add_constrs("depot_balance", [(self._x, out[:1] - into[:1])], Sense.EQ, 0.0)
        builder.add_constrs("fleet_max", [(self._x, out[:1])], Sense.LE, k_max)
        builder.add_constrs("fleet_min", [(self._x, out[:1])], Sense.GE, k_min)
        lb = inst.demand.copy()
        load_ub = np.r_[0.0, np.full(n - 1, inst.capacity)]
        self._load = builder.add_vars("load", n, lb=lb, ub=load_ub)
        i, j = np.nonzero(~np.eye(n, dtype=bool))
        keep = (i > 0) & (j > 0)
        i, j = i[keep], j[keep]
        rows = np.arange(len(i))
        arc = sp.csr_array(
            (np.full(len(i), inst.capacity), (rows, i * n + j)), shape=(len(i), n * n)
        )
        load = sp.csr_array(
            (np.r_[np.ones(len(i)), -np.ones(len(i))], (np.r_[rows, rows], np.r_[i, j])),
            shape=(len(i), n),
        )
        builder.add_constrs(
            "mtz_load",
            [(self._x, arc), (self._load, load)],
            Sense.LE,
            inst.capacity - inst.demand[j],
        )

    # ------------------------------------------------------------ separation
    #: Support thresholds for the fractional component heuristic.
    THRESHOLDS = (EPS, 0.25, 0.5, 0.75)

    def separate(self, x: FloatArray) -> list[Cut]:
        """Violated rounded capacity inequalities on customer components.

        Exact for integer points. For fractional points the components of the
        customer graph are taken at several support thresholds - a cheap,
        standard heuristic (exact RCI separation is NP-hard).
        """
        e, inst, n = self.edge_index, self.data, self.data.n
        values = x[self._x.slice]
        w = e.to_matrix(values)[1:, 1:]
        integral = bool(np.all(np.abs(values - np.rint(values)) < EPS))
        cuts, seen = [], set()
        for threshold in self.THRESHOLDS[:1] if integral else self.THRESHOLDS:
            for comp in components(w, threshold):
                nodes = comp + 1
                key = frozenset(nodes.tolist())
                if key in seen:
                    continue
                seen.add(key)
                need = 2.0 * math.ceil(float(inst.demand[nodes].sum()) / inst.capacity - 1e-9)
                cols = e.delta(mask(n, nodes))
                if values[cols].sum() < need - EPS:
                    cuts.append(Cut(self._x.offset + cols, np.ones(len(cols)), Sense.GE, need))
        return cuts

    def lazy_constraints(self) -> LazyOracle | None:
        """Rounded capacity inequalities for the two-index formulation."""
        return self.separate if self.options.formulation == "two_index" else None

    def user_cuts(self) -> LazyOracle | None:
        """Same separator on fractional root points (Gurobi)."""
        ok = self.options.formulation == "two_index" and self.options.root_cuts
        return self.separate if ok else None

    # ------------------------------------------------------------- warm start
    def heuristic_routes(self) -> list[list[int]]:
        """Clarke-Wright savings routes."""
        return clarke_wright(self.data.dist, self.data.demand, self.data.capacity)

    def warm_start(self) -> FloatArray | None:
        """MIP start from :meth:`heuristic_routes` if it respects the fleet size."""
        if self.options.warm_start == "none":
            return None
        routes = self.heuristic_routes()
        k_min, k_max = self._depot_bounds()
        if not k_min <= len(routes) <= k_max:
            return None
        start = np.zeros(self.build().num_vars)
        n = self.data.n
        for route in routes:
            path = [0, *route, 0]
            for a, b in pairwise(path):
                if self.options.formulation == "two_index":
                    start[self._x.offset + self.edge_index.id(a, b)] += 1.0
                else:
                    start[self._x.offset + a * n + b] = 1.0
            if self._load is not None:
                cumulative = np.cumsum(self.data.demand[route])
                start[self._load.offset + np.asarray(route)] = cumulative
        return start

    def reference_objective(self) -> float | None:
        """Published optimum (CVRPLIB ``.sol``), if known."""
        return self.data.reference

    @classmethod
    def variants(cls) -> dict[str, dict[str, Any]]:
        """Formulations compared by ``linopt variants``."""
        return {
            "two-index + cuts": {"formulation": "two_index"},
            "mtz load": {"formulation": "mtz"},
        }

    # ------------------------------------------------------------- solution
    def _routes(self, values: FloatArray) -> list[list[int]]:
        n = self.data.n
        if self.options.formulation == "two_index":
            mult = np.rint(self.edge_index.to_matrix(values)).astype(np.int64)
            routes = []
            while mult[0].sum() > 0:
                cur = int(np.flatnonzero(mult[0])[0])
                mult[0, cur] -= 1
                mult[cur, 0] -= 1
                route = [cur]
                while True:
                    nxt = int(np.argmax(mult[cur]))
                    if mult[cur, nxt] <= 0:
                        raise ValueError("edge solution does not decompose into routes")
                    mult[cur, nxt] -= 1
                    mult[nxt, cur] -= 1
                    if nxt == 0:
                        break
                    route.append(nxt)
                    cur = nxt
                routes.append(route)
            return routes
        arcs = values.reshape(n, n) > 0.5
        succ = {a: int(np.argmax(arcs[a])) for a in range(1, n)}
        routes = []
        for first in np.flatnonzero(arcs[0]):
            route, cur = [int(first)], int(first)
            while succ[cur] != 0:
                cur = succ[cur]
                route.append(cur)
                if len(route) > n:
                    raise ValueError("arc solution contains a cycle")
            routes.append(route)
        return routes

    def _extract(self, raw: RawSolution) -> CVRPSolution:
        assert raw.x is not None
        assert raw.objective is not None
        routes = self._routes(raw.x[self._x.slice])
        d = self.data
        return CVRPSolution(
            routes=tuple(tuple(r) for r in routes),
            loads=tuple(float(d.demand[r].sum()) for r in routes),
            costs=tuple(d.route_cost(r) for r in routes),
            objective=raw.objective,
            progress=raw.progress,
        )

    def validate(self, solution: CVRPSolution) -> list[str]:
        """Every customer once, capacities, fleet size and cost - from raw data."""
        inst, issues = self.data, []
        visited = sorted(v for r in solution.routes for v in r)
        if visited != list(range(1, inst.n)):
            issues.append("customers are not visited exactly once")
        for k, (route, load) in enumerate(zip(solution.routes, solution.loads, strict=True)):
            if load > inst.capacity + EPS:
                issues.append(
                    f"route {k + 1} ({route[:3]}...) carries {load:g} > {inst.capacity:g}"
                )
        k_min, k_max = self._depot_bounds()
        if not k_min <= len(solution.routes) <= k_max:
            issues.append(f"{len(solution.routes)} routes, expected {k_min:g}..{k_max:g}")
        if not np.isclose(solution.total_cost, solution.objective, rtol=1e-6, atol=1e-6):
            issues.append(
                f"objective {solution.objective:.6f} != route costs {solution.total_cost:.6f}"
            )
        return issues
