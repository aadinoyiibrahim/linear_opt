from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from linear_opt.core.config import SolverSettings, load_config
from linear_opt.core.study import run_variants
from linear_opt.data import jsplib
from linear_opt.problems import registry
from linear_opt.problems.jobshop import (
    JobShopInstance,
    JobShopModel,
    best_heuristic,
    critical_path,
    giffler_thompson,
    left_shift,
    machine_sequences,
    schedule_violations,
)
from linear_opt.solvers import SolverBackend

from .conftest import CONFIG_DIR, satisfies

EXACT = SolverSettings(time_limit=60, mip_gap=0.0)

TOY = """# comment line
3 2
0 3 1 2
1 2 0 1
0 2 1 4
"""

ORLIB = """ +++
 instance toy
 +++
 a made-up instance
 3 2
 0 3 1 2
 1 2 0 1
 0 2 1 4
 +++
 instance other
 2 1
 0 5
 0 6
 +++
"""


def toy() -> JobShopInstance:
    return JobShopInstance.from_data(jsplib.parse_jsp(TOY, "toy"))


# ----------------------------------------------------------------- parsing
def test_parse_standard_format() -> None:
    data = jsplib.parse_jsp(TOY, "toy")
    assert data.machines.tolist() == [[0, 1], [1, 0], [0, 1]]
    assert data.durations.tolist() == [[3, 2], [2, 1], [2, 4]]


@pytest.mark.parametrize(
    ("text", "message"),
    [("", "empty"), ("2 2\n0 1 1 1\n", "expected 8"), ("1 2\n0 1 5 1\n", "machine index")],
)
def test_parse_errors(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        jsplib.parse_jsp(text)


def test_parse_orlib_multi_instance_file() -> None:
    found = jsplib.parse_orlib_jobshop(ORLIB)
    assert set(found) == {"toy", "other"}
    assert found["toy"].durations.tolist() == [[3, 2], [2, 1], [2, 4]]
    assert found["other"].machines.shape == (2, 1)


def test_reference_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    meta = [
        {"name": "ft06", "optimum": 55},
        {"name": "open1", "optimum": None, "bounds": {"lower": 10, "upper": 12}},
    ]
    monkeypatch.setattr(jsplib, "_metadata", lambda: meta)
    assert jsplib.reference("ft06") == jsplib.Reference(55.0)
    assert jsplib.reference("open1") == jsplib.Reference(None, 10.0, 12.0)
    assert jsplib.reference("nope") is None


def test_snapshot_from_jsplib_and_orlib(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_fetch(url: str, filename: str, **kwargs: object) -> Path:
        path = tmp_path / filename
        path.write_text(
            json.dumps([{"name": "toy", "optimum": 7}]) if url.endswith(".json") else TOY
        )
        return path

    monkeypatch.setattr(jsplib, "fetch", fake_fetch)
    written = jsplib.snapshot(["toy"], tmp_path / "fx")
    assert sorted(p.name for p in written) == ["instances.json", "toy"]
    orlib_file = tmp_path / "jobshop1.txt"
    orlib_file.write_text(ORLIB)
    (out,) = jsplib.snapshot(["other"], tmp_path / "fx2", orlib_file=orlib_file)
    assert jsplib.parse_jsp(out.read_text()).durations.tolist() == [[5], [6]]
    with pytest.raises(ValueError, match="not in"):
        jsplib.snapshot(["missing"], tmp_path / "fx3", orlib_file=orlib_file)


# ---------------------------------------------------------------- instance
def test_instance_quantities() -> None:
    inst = toy()
    assert (inst.n_jobs, inst.n_ops, inst.n_machines) == (3, 2, 2)
    np.testing.assert_array_equal(inst.heads, [0, 3, 0, 2, 0, 2])
    np.testing.assert_array_equal(inst.tails, [2, 0, 1, 0, 4, 0])
    assert inst.lower_bound == 8  # machine loads 3+1+2 = 6 and 2+2+4 = 8; longest job 6
    assert len(inst.machine_pairs()) == 6  # 3 ops per machine -> 3 pairs each
    assert inst.label(3) == "J2·2"
    with pytest.raises(ValueError, match="equal shape"):
        JobShopInstance("bad", np.zeros((2, 2), dtype=np.int64), np.zeros((2, 3)))


# -------------------------------------------------------------- heuristics
@settings(max_examples=40, deadline=None)
@given(n=st.integers(1, 8), m=st.integers(1, 6), seed=st.integers(0, 10**6))
def test_giffler_thompson_and_left_shift(n: int, m: int, seed: int) -> None:
    inst = JobShopInstance.random(n, m, seed=seed)
    for rule in ("mwkr", "spt"):
        starts = giffler_thompson(inst, rule)
        assert schedule_violations(inst, starts) == []
        makespan = float((starts + inst.durations).max())
        assert makespan >= inst.lower_bound - 1e-9
        shifted = left_shift(inst, starts)
        assert schedule_violations(inst, shifted) == []
        assert float((shifted + inst.durations).max()) <= makespan + 1e-9
        np.testing.assert_allclose(left_shift(inst, shifted), shifted)  # idempotent
        assert machine_sequences(inst, shifted) == machine_sequences(inst, starts)
        path = critical_path(inst, shifted)
        s = shifted.ravel()
        assert s[path[0]] == 0  # a left-shifted schedule has a critical path from time 0
        assert inst.p[path].sum() == pytest.approx(float((shifted + inst.durations).max()))


def test_left_shift_repairs_tolerance_overlaps() -> None:
    inst = toy()
    starts = left_shift(inst, best_heuristic(inst))
    jittered = starts + np.array([[0, 0], [0, -2e-4], [0, 0]])  # tiny overlap
    assert schedule_violations(inst, jittered)
    assert schedule_violations(inst, left_shift(inst, jittered)) == []


# ------------------------------------------------------------------- model
FORMS = [
    {"formulation": "disjunctive", "big_m": "tight"},
    {"formulation": "disjunctive", "big_m": "naive"},
    {"formulation": "time_indexed"},
]


@pytest.mark.parametrize("options", FORMS, ids=lambda o: "-".join(o.values()))
def test_formulations_reach_the_same_optimum(
    backend: SolverBackend, options: dict[str, str]
) -> None:
    inst = JobShopInstance.random(4, 3, seed=2, max_duration=6)
    reference = JobShopModel(inst).solve(EXACT).raw.objective
    result = JobShopModel(inst, options).solve(EXACT, backend=backend)
    assert result.ok, result.violations
    assert result.raw.objective == pytest.approx(reference)
    assert result.solution is not None
    assert result.solution.makespan == pytest.approx(reference)


@pytest.mark.parametrize("options", FORMS, ids=lambda o: "-".join(o.values()))
def test_warm_start_is_feasible(options: dict[str, str]) -> None:
    model = JobShopModel(JobShopInstance.random(5, 3, seed=4, max_duration=5), options)
    start = model.warm_start()
    assert start is not None
    assert satisfies(model.build(), start) == []


def test_relaxations_are_valid_lower_bounds(backend: SolverBackend) -> None:
    inst = JobShopInstance.random(5, 3, seed=7, max_duration=6)
    rows = run_variants(JobShopModel(inst), EXACT, backend)
    best = rows[0].result.raw.objective
    assert best is not None
    for r in rows:
        assert r.lp_bound is not None
        assert r.lp_bound <= best + 1e-6
        assert r.result.raw.objective == pytest.approx(best)
    longest_job = float(inst.durations.sum(axis=1).max())
    lp = rows[0].lp_bound
    assert lp is not None
    assert lp >= longest_job - 1e-6  # implied by job precedences


def test_time_indexed_requires_integer_durations() -> None:
    inst = JobShopInstance("frac", np.array([[0]], dtype=np.int64), np.array([[1.5]]))
    with pytest.raises(ValueError, match="integer"):
        JobShopModel(inst, {"formulation": "time_indexed"}).build()


def test_validate_catches_overlaps() -> None:
    inst = toy()
    starts = np.zeros((3, 2))
    issues = schedule_violations(inst, starts, objective=1.0)
    assert any("overlaps" in i for i in issues)
    assert any("starts before" in i for i in issues)
    assert any("objective" in i for i in issues)


def test_registry_builds_jobshop_models(tmp_path: Path) -> None:
    model = registry.model_from_config(load_config(CONFIG_DIR / "jobshop_synthetic.toml"))
    assert isinstance(model, JobShopModel) and model.data.n_jobs == 6
    path = tmp_path / "toy.jsp"
    path.write_text(TOY)
    cfg = load_config(CONFIG_DIR / "jobshop_synthetic.toml")
    cfg = cfg.model_copy(update={"instance": cfg.instance.model_validate({"path": path})})
    assert registry.model_from_config(cfg).data.n_jobs == 3


# ---------------------------------------------------------- JSPLIB oracles
def _has(name: str) -> bool:
    return (jsplib.fixture_dir() / name).exists() and (
        jsplib.fixture_dir() / "instances.json"
    ).exists()


@pytest.mark.skipif(not _has("ft06"), reason="run `linopt data jsplib` first")
@pytest.mark.parametrize("options", FORMS, ids=lambda o: "-".join(o.values()))
def test_ft06_reaches_the_published_optimum(
    backend: SolverBackend, options: dict[str, str]
) -> None:
    model = registry.model_from_config(load_config(CONFIG_DIR / "jobshop_ft06.toml"))
    result = model.with_options(**options).solve(EXACT, backend=backend)
    assert result.ok
    assert result.reference == 55
    assert result.raw.objective == pytest.approx(55)


@pytest.mark.skipif(not _has("la01"), reason="run `linopt data jsplib` first")
@pytest.mark.parametrize("name", ["la01", "la02", "la03", "la04", "la05"])
def test_lawrence_instances_reach_published_optima(name: str, gurobi: SolverBackend) -> None:
    data, ref = jsplib.load_jsp(name)
    assert ref is not None and ref.optimum is not None
    result = JobShopModel(JobShopInstance.from_data(data, ref)).solve(EXACT, backend=gurobi)
    assert result.ok
    assert result.raw.objective == pytest.approx(ref.optimum)
