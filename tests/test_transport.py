from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from linear_opt.core.config import BackendName, SolverSettings, load_config
from linear_opt.core.duality import dual_sign_violations, duality_gap
from linear_opt.core.result import SolveStatus
from linear_opt.data import geonames
from linear_opt.data.geonames import City
from linear_opt.problems import registry
from linear_opt.problems.transport import TransportInstance, TransportModel
from linear_opt.solvers import SolverBackend

from .conftest import CONFIG_DIR, INSTALLED

FAST = SolverSettings(time_limit=10)


def solve(inst: TransportInstance, backend: SolverBackend, **options: str):  # type: ignore[no-untyped-def]
    return TransportModel(inst, options).solve(FAST, backend=backend)


def test_solution_is_feasible_and_certified_by_duality(backend: SolverBackend) -> None:
    model = TransportModel(TransportInstance.random(6, 15, seed=1))
    result = model.solve(FAST, backend=backend)
    assert result.ok, result.violations
    sol, lm = result.solution, model.build()
    assert sol is not None and result.raw.duals is not None
    assert duality_gap(lm, sol.total_cost, result.raw.duals) < 1e-9
    assert dual_sign_violations(lm, result.raw.duals) == []
    assert sol.demand_duals is not None and np.all(sol.demand_duals >= -1e-9)
    assert sol.supply_duals is not None and np.all(sol.supply_duals <= 1e-9)
    # Complementary slackness: used routes have zero reduced cost.
    assert sol.reduced_costs is not None
    assert np.all(np.abs(sol.reduced_costs[sol.flow > 1e-7]) < 1e-7)


@pytest.mark.skipif(len(INSTALLED) < 2, reason="needs both backends")
def test_backends_agree() -> None:
    inst = TransportInstance.random(10, 25, seed=3)
    values = [TransportModel(inst).solve(FAST, backend=b).raw.objective for b in INSTALLED]
    assert values[0] == pytest.approx(values[1], rel=1e-9)


@settings(
    max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(seed=st.integers(0, 10_000), shift=st.floats(0.0, 5.0), col=st.integers(0, 7))
def test_column_cost_shift_moves_optimum_by_shift_times_demand(
    backend: SolverBackend, seed: int, shift: float, col: int
) -> None:
    """OT invariance: adding k to every cost into sink j adds exactly k * d_j."""
    base = TransportInstance.random(4, 8, seed=seed)
    cost = base.cost.copy()
    cost[:, col] += shift
    shifted = TransportInstance(
        base.sources, base.sinks, base.supply, base.demand, cost, base.source_xy, base.sink_xy
    )
    a, b = solve(base, backend).raw.objective, solve(shifted, backend).raw.objective
    assert b - a == pytest.approx(shift * base.demand[col], rel=1e-7, abs=1e-7)


@settings(
    max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(seed=st.integers(0, 10_000), alpha=st.floats(0.1, 100.0))
def test_cost_scaling_scales_optimum(backend: SolverBackend, seed: int, alpha: float) -> None:
    base = TransportInstance.random(4, 6, seed=seed)
    scaled = TransportInstance(
        base.sources, base.sinks, base.supply, base.demand, alpha * base.cost,
        base.source_xy, base.sink_xy,
    )  # fmt: skip
    a, b = solve(base, backend).raw.objective, solve(scaled, backend).raw.objective
    assert b == pytest.approx(alpha * a, rel=1e-7)


def test_balanced_instance_with_equality_supply(backend: SolverBackend) -> None:
    inst = TransportInstance.random(5, 9, seed=4, slack=0.0)
    assert inst.is_balanced
    result = solve(inst, backend, supply_constraint="eq")
    assert result.ok
    assert result.solution is not None
    np.testing.assert_allclose(result.solution.flow.sum(axis=1), inst.supply, rtol=1e-7)


def _short_supply() -> TransportInstance:
    base = TransportInstance.random(3, 4, seed=0)
    return TransportInstance(
        base.sources, base.sinks, base.supply * 0.5, base.demand, base.cost,
        base.source_xy, base.sink_xy,
    )  # fmt: skip


def test_infeasible_when_demand_exceeds_supply(backend: SolverBackend) -> None:
    result = solve(_short_supply(), backend)
    assert result.raw.status in {SolveStatus.INFEASIBLE, SolveStatus.INF_OR_UNBD}
    assert result.solution is None and not result.ok


def test_gurobi_iis_explains_shortage(gurobi: SolverBackend) -> None:
    model = TransportModel(_short_supply())
    iis = gurobi.compute_iis(model.build(), FAST)
    assert any(n.startswith("supply[") for n in iis)
    assert any(n.startswith("demand[") for n in iis)


def test_validate_catches_tampered_solution(backend: SolverBackend) -> None:
    model = TransportModel(TransportInstance.random(3, 5, seed=2))
    sol = model.solve(FAST, backend=backend).solution
    assert sol is not None
    from dataclasses import replace

    bad = replace(sol, flow=sol.flow * 0.5)
    issues = model.validate(bad)
    assert any("demand" in i for i in issues)
    assert any("objective" in i for i in issues)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("supply", np.ones(2), "supply has shape"),
        ("demand", -np.ones(4), "non-negative"),
        ("cost", np.full((3, 4), np.nan), "finite"),
    ],
)
def test_instance_validation(field: str, value: np.ndarray, message: str) -> None:
    base = TransportInstance.random(3, 4, seed=0)
    kwargs = {
        k: getattr(base, k)
        for k in ("sources", "sinks", "supply", "demand", "cost", "source_xy", "sink_xy")
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        TransportInstance(**kwargs)


CITIES = [
    City(1, "Alpha", 52.52, 13.40, 3_600_000),
    City(2, "Beta", 53.55, 9.99, 1_800_000),
    City(3, "Gamma", 48.14, 11.58, 1_500_000),
    City(4, "Delta", 50.94, 6.96, 1_000_000),
    City(5, "Epsilon", 50.11, 8.68, 750_000),
]


def test_from_cities_splits_and_scales() -> None:
    inst = TransportInstance.from_cities(CITIES, n_sources=2, seed=0, slack=0.2)
    assert inst.shape == (2, 3) and inst.geographic and inst.cost_unit == "km"
    assert inst.supply.sum() == pytest.approx(1.2 * inst.demand.sum())
    assert set(inst.sources) | set(inst.sinks) == {c.name for c in CITIES}
    assert np.all(inst.cost > 50)  # distinct German-scale cities are > 50 km apart


@pytest.mark.parametrize(("n_sources", "slack"), [(0, 0.1), (5, 0.1), (2, -0.1)])
def test_from_cities_rejects_bad_parameters(n_sources: int, slack: float) -> None:
    with pytest.raises(ValueError, match=r"n_sources|slack"):
        TransportInstance.from_cities(CITIES, n_sources=n_sources, slack=slack)


def test_registry_builds_synthetic_model() -> None:
    cfg = load_config(CONFIG_DIR / "transport_synthetic.toml")
    model = registry.model_from_config(cfg)
    assert isinstance(model, TransportModel)
    params = cfg.instance.params
    assert model.data.shape == (params["n_sources"], params["n_sinks"])


def test_registry_reports_missing_city_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(geonames, "fixture_path", lambda: tmp_path / "missing.csv")
    cfg = load_config(CONFIG_DIR / "transport_de_cities.toml")
    with pytest.raises(FileNotFoundError, match="linopt data cities"):
        registry.model_from_config(cfg)


def test_registry_uses_city_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(geonames, "load_fixture", lambda: CITIES)
    cfg = load_config(CONFIG_DIR / "transport_de_cities.toml")
    cfg = cfg.model_copy(
        update={
            "instance": cfg.instance.model_copy(update={"params": {"n_cities": 5, "n_sources": 2}})
        }
    )
    model = registry.model_from_config(cfg)
    assert model.data.shape == (2, 3)


@pytest.mark.parametrize(
    "config",
    [
        "transport_synthetic",
        "facility_synthetic",
        "tsp_synthetic",
        "cvrp_synthetic",
        "jobshop_synthetic",
    ],
)
def test_registry_rejects_unknown_sources(config: str) -> None:
    cfg = load_config(CONFIG_DIR / f"{config}.toml")
    cfg = cfg.model_copy(update={"instance": cfg.instance.model_copy(update={"source": "nowhere"})})
    with pytest.raises(ValueError, match="unsupported instance source"):
        registry.model_from_config(cfg)


@pytest.mark.skipif(BackendName.HIGHS not in INSTALLED, reason="needs HiGHS")
def test_model_accepts_backend_name() -> None:
    result = TransportModel(TransportInstance.random(2, 3)).solve(backend="highs")
    assert result.raw.backend == "highs"
