from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from linear_opt.core.config import SolverSettings, load_config
from linear_opt.core.study import run_variants
from linear_opt.data import tsplib
from linear_opt.problems import registry
from linear_opt.problems.tsp import (
    TSPInstance,
    TSPModel,
    tour_from_edges,
    tour_from_successors,
)
from linear_opt.solvers import SolverBackend

from .conftest import CONFIG_DIR, INSTALLED, satisfies
from .test_transport import CITIES

EXACT = SolverSettings(time_limit=60, mip_gap=0.0)
FORMULATIONS = ["dfj", "scf", "mtz"]


@pytest.mark.parametrize("formulation", FORMULATIONS)
def test_formulations_reach_the_same_optimum(backend: SolverBackend, formulation: str) -> None:
    inst = TSPInstance.random(12, seed=1)
    reference = TSPModel(inst, {"formulation": "dfj"}).solve(EXACT).raw.objective
    result = TSPModel(inst, {"formulation": formulation}).solve(EXACT, backend=backend)
    assert result.ok, result.violations
    assert result.raw.objective == pytest.approx(reference)
    assert result.solution is not None
    assert len(result.solution.tour) == 12


@pytest.mark.skipif(len(INSTALLED) < 2, reason="needs both backends")
def test_backends_agree_on_larger_instance() -> None:
    inst = TSPInstance.random(40, seed=8)
    values = [TSPModel(inst).solve(EXACT, backend=b).raw.objective for b in INSTALLED]
    assert values[0] == pytest.approx(values[1])


@settings(
    max_examples=6, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(seed=st.integers(0, 10_000))
def test_lp_bounds_are_ordered(backend: SolverBackend, seed: int) -> None:
    """Padberg & Sung (1991): z_LP(MTZ) <= z_LP(flow) <= z_LP(DFJ) <= z*."""
    rows = {
        r.name: r for r in run_variants(TSPModel(TSPInstance.random(10, seed=seed)), EXACT, backend)
    }
    mtz, scf, dfj = (rows[k].lp_bound for k in ("mtz", "single-commodity flow", "dfj (lazy)"))
    assert mtz is not None and scf is not None and dfj is not None
    best = rows["dfj (lazy)"].result.raw.objective
    assert best is not None
    assert mtz <= scf + 1e-6 <= dfj + 2e-6 <= best + 3e-6


def test_separation_on_integer_points() -> None:
    model = TSPModel(TSPInstance.random(6, seed=0))
    lm = model.build()
    e = model.edge_index
    x = np.zeros(lm.num_vars)
    for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:  # two triangles
        x[e.id(a, b)] = 1
    cuts = model.separate(x)
    assert len(cuts) == 1  # S and its complement give the same cut
    assert cuts[0].violation(x) == pytest.approx(2.0)
    assert model.history[-1].n_subtours == 2
    tour = np.zeros(lm.num_vars)
    for k in range(6):
        tour[e.id(k, (k + 1) % 6)] = 1
    assert model.separate(tour) == []


def test_separation_on_fractional_points_uses_min_cut() -> None:
    model = TSPModel(TSPInstance.random(6, seed=0))
    e = model.edge_index
    x = np.zeros(model.build().num_vars)
    for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        x[e.id(a, b)] = 0.9
    x[e.id(2, 3)] = x[e.id(0, 5)] = 0.1  # connected, but the cut {0,1,2} has weight 0.2
    cuts = model.separate(x, record=False)
    assert cuts
    assert max(c.violation(x) for c in cuts) == pytest.approx(1.8)


@pytest.mark.parametrize("formulation", FORMULATIONS)
def test_warm_start_is_feasible(formulation: str) -> None:
    model = TSPModel(TSPInstance.random(9, seed=4), {"formulation": formulation})
    start = model.warm_start()
    assert start is not None
    assert satisfies(model.build(), start) == []
    assert TSPModel(model.data, {"warm_start": "none"}).warm_start() is None


def test_tour_helpers() -> None:
    assert tour_from_successors({0: 2, 2: 1, 1: 0}) == [0, 2, 1]
    assert tour_from_edges(4, [(0, 1), (1, 2), (2, 3), (0, 3)]) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="single tour"):
        tour_from_successors({0: 1, 1: 0, 2: 3, 3: 2, 4: 4})
    with pytest.raises(ValueError, match="Hamiltonian"):
        tour_from_edges(6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)])


def test_instance_validation_and_cities() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        TSPInstance(("a", "b"), np.zeros((2, 2)))
    asym = np.array([[0, 1, 2], [3, 0, 1], [2, 1, 0]], dtype=float)
    with pytest.raises(ValueError, match="symmetric"):
        TSPInstance(("a", "b", "c"), asym)
    inst = TSPInstance.from_cities(CITIES)
    assert inst.geographic and inst.unit == "km" and inst.n == 5
    assert inst.dist[0, 2] == pytest.approx(504, abs=3)  # Berlin-Munich in the toy list


def test_registry_builds_tsp_models() -> None:
    model = registry.model_from_config(load_config(CONFIG_DIR / "tsp_synthetic.toml"))
    assert isinstance(model, TSPModel) and model.data.n == 25


# ---------------------------------------------------------- TSPLIB oracles
def _has(name: str) -> bool:
    return (tsplib.fixture_dir("tsplib") / f"{name}.tsp").exists()


@pytest.mark.parametrize("name", sorted(tsplib.TSP_OPTIMA))
def test_published_optimum_matches_optimal_tour(name: str) -> None:
    """Checks our distance functions (EUC_2D, ATT, GEO, EXPLICIT) against TSPLIB."""
    if not _has(name):
        pytest.skip(f"run `linopt data tsplib {name}` first")
    if not (tsplib.fixture_dir("tsplib") / f"{name}.opt.tour").exists():
        pytest.skip(f"TSPLIB publishes no optimal tour for {name} (solved test covers it)")
    problem, optimum, tour = tsplib.load_tsplib(name)
    assert tour is not None
    assert tsplib.tour_length(problem.distances, tour) == optimum


@pytest.mark.parametrize("name", ["burma14", "gr17", "dantzig42", "ulysses22"])
def test_instances_without_published_tour_reach_the_published_optimum(
    name: str, backend: SolverBackend
) -> None:
    """GEO and EXPLICIT instances: compare the solved optimum with TSPLIB's table."""
    if not _has(name):
        pytest.skip(f"run `linopt data tsplib {name}` first")
    problem, optimum, tour = tsplib.load_tsplib(name)
    result = TSPModel(TSPInstance.from_tsplib(problem, optimum, tour)).solve(EXACT, backend=backend)
    assert result.ok
    assert result.raw.objective == pytest.approx(optimum)


@pytest.mark.skipif(not _has("berlin52"), reason="run `linopt data tsplib berlin52` first")
def test_berlin52_is_solved_to_the_published_optimum(backend: SolverBackend) -> None:
    model = registry.model_from_config(load_config(CONFIG_DIR / "tsp_berlin52.toml"))
    result = model.solve(SolverSettings(time_limit=300, mip_gap=0.0), backend=backend)
    assert result.ok and result.raw.objective == pytest.approx(7542)


def test_reference_comes_from_instance(tmp_path: Path) -> None:
    inst = TSPInstance.random(5, seed=0)
    assert TSPModel(inst).reference_objective() is None


@pytest.mark.large
@pytest.mark.parametrize("name", ["st70", "eil76", "kroA100"])
def test_large_tsplib_instances(name: str, gurobi: SolverBackend) -> None:
    """Above the restricted licence (2415-4950 edge variables): needs a full Gurobi licence."""
    if not _has(name):
        pytest.skip(f"run `linopt data tsplib {name}` first")
    problem, optimum, tour = tsplib.load_tsplib(name)
    model = TSPModel(TSPInstance.from_tsplib(problem, optimum, tour))
    result = model.solve(SolverSettings(time_limit=600, mip_gap=0.0), backend=gurobi)
    assert result.ok
    assert result.raw.objective == pytest.approx(optimum)
