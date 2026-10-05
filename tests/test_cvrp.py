from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from linear_opt.core.config import SolverSettings, load_config
from linear_opt.data import tsplib
from linear_opt.problems import registry
from linear_opt.problems.cvrp import CVRPInstance, CVRPModel, CVRPSolution
from linear_opt.solvers import SolverBackend

from .conftest import CONFIG_DIR, INSTALLED, satisfies
from .test_transport import CITIES
from .test_tsplib import CVRP as TOY_VRP

EXACT = SolverSettings(time_limit=60, mip_gap=0.0)


def inst(seed: int = 5, n: int = 10) -> CVRPInstance:
    return CVRPInstance.random(n, seed=seed, capacity=60)


@pytest.mark.parametrize("formulation", ["two_index", "mtz"])
def test_formulations_agree(backend: SolverBackend, formulation: str) -> None:
    data = inst()
    reference = CVRPModel(data).solve(EXACT).raw.objective
    result = CVRPModel(data, {"formulation": formulation}).solve(EXACT, backend=backend)
    assert result.ok, result.violations
    assert result.raw.objective == pytest.approx(reference)


@pytest.mark.skipif(len(INSTALLED) < 2, reason="needs both backends")
def test_backends_agree() -> None:
    data = inst(seed=6, n=11)
    values = [CVRPModel(data).solve(EXACT, backend=b).raw.objective for b in INSTALLED]
    assert values[0] == pytest.approx(values[1])


def test_exact_fleet_uses_all_vehicles(backend: SolverBackend) -> None:
    data = inst()
    free = CVRPModel(data).solve(EXACT, backend=backend)
    exact = CVRPModel(data, {"fleet": "exact"}).solve(EXACT, backend=backend)
    assert exact.ok, exact.violations
    assert exact.solution is not None and len(exact.solution.routes) == data.vehicles
    assert exact.raw.objective is not None and free.raw.objective is not None
    assert exact.raw.objective >= free.raw.objective - 1e-9


def test_separation_detects_overload_and_subtours() -> None:
    data = CVRPInstance.random(4, seed=0, capacity=100)
    data = replace(data, demand=np.array([0.0, 40, 40, 40, 40]), vehicles=3)
    model = CVRPModel(data)
    e = model.edge_index
    x = np.zeros(model.build().num_vars)
    for a, b in [(0, 1), (1, 2), (2, 3), (0, 3)]:  # route 1-2-3 carries 120 > 100
        x[e.id(a, b)] += 1
    x[e.id(0, 4)] = 2  # out-and-back to customer 4
    cuts = model.separate(x)
    assert len(cuts) == 1 and cuts[0].rhs == 4.0  # 2 * ceil(120 / 100)
    assert cuts[0].violation(x) == pytest.approx(2.0)
    y = np.zeros_like(x)
    for a, b in [(1, 2), (2, 3), (1, 3)]:  # subtour away from the depot
        y[e.id(a, b)] = 1
    y[e.id(0, 4)] = 2
    assert any(c.violation(y) > 0 for c in model.separate(y))


@pytest.mark.parametrize("formulation", ["two_index", "mtz"])
def test_savings_warm_start_is_feasible(formulation: str) -> None:
    model = CVRPModel(inst(n=12), {"formulation": formulation})
    start = model.warm_start()
    assert start is not None
    assert satisfies(model.build(), start) == []


def test_validate_catches_bad_solutions() -> None:
    model = CVRPModel(inst())
    sol = model.solve(EXACT).solution
    assert sol is not None
    merged = CVRPSolution(
        (sum(sol.routes, ()),), (sum(sol.loads),), (sol.total_cost,), sol.objective
    )
    issues = model.validate(merged)
    assert any("carries" in i for i in issues)
    assert any("routes, expected" in i for i in issues)


def test_instance_validation() -> None:
    data = inst()
    with pytest.raises(ValueError, match="capacity"):
        replace(data, demand=np.r_[0.0, np.full(data.n - 1, 61.0)])
    with pytest.raises(ValueError, match="vehicles"):
        replace(data, vehicles=1)
    with pytest.raises(ValueError, match="depot demand"):
        replace(data, demand=np.r_[1.0, data.demand[1:]])


def test_from_tsplib_puts_depot_first() -> None:
    p = tsplib.parse_tsplib(TOY_VRP.replace(" 1\n -1", " 3\n -1"))
    p = replace(p, demand=np.array([6.0, 6.0, 0.0, 3.0]))
    data = CVRPInstance.from_tsplib(p)
    assert data.names[0] == "3" and data.demand[0] == 0 and data.vehicles == 2


def test_from_cities_and_registry() -> None:
    data = CVRPInstance.from_cities(CITIES, depot="Gamma", capacity=200, vehicles=3)
    assert data.names[0] == "Gamma" and data.geographic
    with pytest.raises(ValueError, match="depot"):
        CVRPInstance.from_cities(CITIES, depot="Nowhere", capacity=200, vehicles=3)
    model = registry.model_from_config(load_config(CONFIG_DIR / "cvrp_synthetic.toml"))
    assert isinstance(model, CVRPModel)


# --------------------------------------------------------- CVRPLIB oracle
HAS_A32 = (tsplib.fixture_dir("cvrplib") / "A-n32-k5.vrp").exists()


@pytest.mark.skipif(not HAS_A32, reason="run `linopt data cvrplib A-n32-k5` first")
def test_reference_routes_reproduce_published_cost() -> None:
    """Checks our reading of the .vrp distances and the .sol node numbering."""
    problem, cost, routes = tsplib.load_cvrplib("A-n32-k5")
    data = CVRPInstance.from_tsplib(problem, cost, routes)
    assert routes is not None and cost is not None
    assert sum(data.route_cost(r) for r in routes) == pytest.approx(cost)
    assert all(data.demand[r].sum() <= data.capacity for r in routes)


@pytest.mark.skipif(not HAS_A32, reason="run `linopt data cvrplib A-n32-k5` first")
def test_a_n32_k5_is_solved_to_the_published_optimum(gurobi: SolverBackend) -> None:
    model = registry.model_from_config(load_config(CONFIG_DIR / "cvrp_a_n32_k5.toml"))
    result = model.solve(SolverSettings(time_limit=600, mip_gap=0.0), backend=gurobi)
    assert result.ok and result.raw.objective == pytest.approx(784)
