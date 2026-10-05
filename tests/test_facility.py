from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from linear_opt.core.config import SolverSettings, load_config
from linear_opt.core.result import SolveStatus
from linear_opt.core.study import run_variants
from linear_opt.data import orlib
from linear_opt.problems import registry
from linear_opt.problems.facility import (
    FacilityInstance,
    FacilityModel,
    FacilitySolution,
    greedy_solution,
)
from linear_opt.solvers import SolverBackend

from .conftest import CONFIG_DIR, INSTALLED
from .test_transport import CITIES

EXACT = SolverSettings(time_limit=30, mip_gap=0.0)
HAS_CAP41 = (orlib.fixture_dir() / "cap41.txt").exists() and (
    orlib.fixture_dir() / "capopt.txt"
).exists()


def inst(seed: int = 0, m: int = 8, n: int = 20, ratio: float = 1.5) -> FacilityInstance:
    return FacilityInstance.random(m, n, seed=seed, capacity_ratio=ratio)


def test_solution_is_feasible(backend: SolverBackend) -> None:
    result = FacilityModel(inst()).solve(EXACT, backend=backend)
    assert result.raw.status is SolveStatus.OPTIMAL
    assert result.ok, result.violations
    sol = result.solution
    assert sol is not None
    assert sol.total_cost == pytest.approx(result.raw.objective, rel=1e-7)
    assert sol.open.any()
    assert result.raw.progress  # both backends record MIP progress


@pytest.mark.skipif(len(INSTALLED) < 2, reason="needs both backends")
def test_backends_agree() -> None:
    data = inst(seed=4, m=10, n=30)
    values = [FacilityModel(data).solve(EXACT, backend=b).raw.objective for b in INSTALLED]
    assert values[0] == pytest.approx(values[1], rel=1e-7)


@settings(
    max_examples=8, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(seed=st.integers(0, 10_000))
def test_strong_relaxation_dominates_weak(backend: SolverBackend, seed: int) -> None:
    """Same integer optimum, but the strong LP bound is never below the weak one."""
    rows = {r.name: r for r in run_variants(FacilityModel(inst(seed)), EXACT, backend)}
    strong, weak = rows["strong"], rows["weak"]
    assert strong.result.raw.objective == pytest.approx(weak.result.raw.objective, rel=1e-7)
    assert strong.lp_bound is not None and weak.lp_bound is not None
    assert strong.lp_bound >= weak.lp_bound - 1e-6
    best = strong.result.raw.objective
    assert best is not None
    assert strong.lp_bound <= best + 1e-6  # relaxation is a lower bound


def test_single_sourcing_costs_at_least_as_much(backend: SolverBackend) -> None:
    data = inst(seed=7)
    split = FacilityModel(data).solve(EXACT, backend=backend)
    single = FacilityModel(data, {"single_sourcing": True}).solve(EXACT, backend=backend)
    assert single.ok, single.violations
    assert single.solution is not None
    x = single.solution.assignment
    np.testing.assert_allclose(x, np.round(x), atol=1e-6)
    assert single.raw.objective is not None and split.raw.objective is not None
    assert single.raw.objective >= split.raw.objective - 1e-6


def test_infeasible_when_capacity_too_small(backend: SolverBackend) -> None:
    data = inst()
    tiny = replace(data, capacity=data.capacity * 0.3)
    result = FacilityModel(tiny).solve(EXACT, backend=backend)
    assert result.raw.status in {SolveStatus.INFEASIBLE, SolveStatus.INF_OR_UNBD}


def test_gurobi_iis_points_at_capacity(gurobi: SolverBackend) -> None:
    data = inst()
    model = FacilityModel(
        replace(data, capacity=data.capacity * 0.3), {"aggregate_capacity_cut": False}
    )
    iis = gurobi.compute_iis(model.build(), EXACT)
    assert any(n.startswith("capacity[") for n in iis)


# ------------------------------------------------------------------ heuristic
@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("seed", range(5))
def test_greedy_solution_is_feasible(seed: int, single: bool) -> None:
    data = inst(seed, ratio=2.0)
    found = greedy_solution(data, single_sourcing=single)
    assert found is not None
    is_open, x = found
    model = FacilityModel(data, {"single_sourcing": single})
    fixed = float(data.fixed_cost[is_open].sum())
    serving = float((data.assign_cost * x).sum())
    sol = FacilitySolution(is_open, x, fixed, serving, fixed + serving)
    assert model.validate(sol) == []


def test_greedy_fails_gracefully_when_single_sourcing_impossible() -> None:
    data = inst()
    demand = data.demand.copy()
    demand[0] = data.capacity.max() * 1.01  # no site can host customer 0 alone
    assert greedy_solution(replace(data, demand=demand), single_sourcing=True) is None


def test_warm_start_vector_covers_all_variables() -> None:
    model = FacilityModel(inst())
    start = model.warm_start()
    assert start is not None
    assert start.shape == (model.build().num_vars,)
    assert not np.isnan(start).any()
    assert FacilityModel(inst(), {"warm_start": "none"}).warm_start() is None


def test_validate_detects_tampering() -> None:
    model = FacilityModel(inst())
    sol = model.solve(EXACT).solution
    assert sol is not None
    closed = replace(sol, open=np.zeros_like(sol.open))
    issues = model.validate(closed)
    assert any("closed facility" in i for i in issues)
    issues = model.validate(replace(sol, assignment=sol.assignment * 0.5))
    assert any("served" in i for i in issues)


# ------------------------------------------------------------------ instances
def test_instance_validation() -> None:
    data = inst()
    with pytest.raises(ValueError, match="shape"):
        replace(data, capacity=data.capacity[:-1])
    with pytest.raises(ValueError, match="non-negative"):
        replace(data, fixed_cost=-data.fixed_cost)


def test_from_cities_capacity_scales_with_population() -> None:
    data = FacilityInstance.from_cities(CITIES, n_candidates=3, capacity_ratio=2.0)
    assert data.shape == (3, 5) and data.geographic
    assert data.capacity.sum() == pytest.approx(2.0 * data.total_demand)
    assert np.all(np.diff(data.capacity) <= 0)  # candidates sorted by population
    with pytest.raises(ValueError, match="n_candidates"):
        FacilityInstance.from_cities(CITIES, n_candidates=0)


def test_reference_only_for_splittable_problem() -> None:
    data = replace(inst(), reference=123.0)
    assert FacilityModel(data).reference_objective() == 123.0
    assert FacilityModel(data, {"single_sourcing": True}).reference_objective() is None


def test_registry_builds_synthetic_facility() -> None:
    model = registry.model_from_config(load_config(CONFIG_DIR / "facility_synthetic.toml"))
    assert isinstance(model, FacilityModel)
    assert model.data.shape == (15, 50)


# ---------------------------------------------------------- real benchmarks
@pytest.mark.skipif(not HAS_CAP41, reason="run `linopt data orlib-cap` first")
def test_cap41_matches_published_optimum(backend: SolverBackend) -> None:
    model = registry.model_from_config(load_config(CONFIG_DIR / "facility_cap41.toml"))
    result = model.solve(EXACT, backend=backend)
    assert result.ok, result.violations
    assert result.reference is not None
    assert result.raw.objective == pytest.approx(result.reference, rel=1e-7)


@pytest.mark.large
@pytest.mark.parametrize("name", ["cap131", "cap134"])
def test_large_cap_instances(name: str, backend: SolverBackend) -> None:
    if not (orlib.fixture_dir() / f"{name}.txt").exists():
        pytest.skip(f"run `linopt data orlib-cap {name}` first")
    data = FacilityInstance.from_orlib(orlib.load_cap(name), orlib.known_optimum(name))
    result = FacilityModel(data).solve(SolverSettings(time_limit=600, mip_gap=0.0), backend=backend)
    assert result.raw.objective == pytest.approx(data.reference, rel=1e-7)
