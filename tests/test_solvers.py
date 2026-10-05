"""Backend contract tests: every installed backend must pass all of these."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from linear_opt.core.config import SolverSettings
from linear_opt.core.duality import dual_objective, dual_sign_violations
from linear_opt.core.ir import LinearModel, ModelBuilder, ObjSense, Sense, VarType
from linear_opt.core.result import Cut, SolveStatus
from linear_opt.solvers import Capability, LicenseLimitError, SolverBackend

FAST = SolverSettings(time_limit=10)


def wyndor() -> LinearModel:
    """max 3x + 5y s.t. x <= 4, 2y <= 12, 3x + 2y <= 18 (optimum 36 at (2, 6))."""
    b = ModelBuilder("wyndor", ObjSense.MAXIMIZE)
    x = b.add_vars("x", 2, obj=[3.0, 5.0])
    b.add_constrs("cap", [(x, [[1, 0], [0, 2], [3, 2]])], Sense.LE, [4, 12, 18])
    return b.build()


def knapsack(cap: float = 7.0) -> LinearModel:
    """Values 10, 13, 7, 8; weights 3, 4, 2, 3; optimum 23 (items 0 and 1)."""
    b = ModelBuilder("knapsack", ObjSense.MAXIMIZE)
    x = b.add_vars("take", 4, vtype=VarType.BINARY, obj=[10, 13, 7, 8])
    b.add_constrs("weight", [(x, [[3, 4, 2, 3]])], Sense.LE, cap)
    return b.build()


def test_lp_optimum_duals_and_reduced_costs(backend: SolverBackend) -> None:
    lm = wyndor()
    raw = backend.solve(lm, FAST)
    assert raw.status is SolveStatus.OPTIMAL
    assert raw.objective == pytest.approx(36.0)
    assert raw.x is not None and raw.duals is not None and raw.reduced_costs is not None
    np.testing.assert_allclose(raw.x, [2, 6], atol=1e-8)
    np.testing.assert_allclose(raw.duals, [0.0, 1.5, 1.0], atol=1e-8)  # textbook shadow prices
    np.testing.assert_allclose(raw.reduced_costs, lm.c - lm.A.T @ raw.duals, atol=1e-8)
    assert dual_objective(lm, raw.duals) == pytest.approx(36.0)
    assert dual_sign_violations(lm, raw.duals) == []
    assert raw.gap == pytest.approx(0.0)


def test_minimisation_with_ge_rows_and_free_variable(backend: SolverBackend) -> None:
    # min x0 + x1 - z  s.t. x0 >= 1, 2 x1 >= 2, z <= 3 (z free)  -> optimum -1
    b = ModelBuilder("mixed")
    x = b.add_vars("x", 2, obj=1.0)
    z = b.add_vars("z", 1, lb=-np.inf, obj=-1.0)
    b.add_constrs("lo", [(x, [[1, 0], [0, 2]])], Sense.GE, [1, 2])
    b.add_constrs("hi", [(z, [[1.0]])], Sense.LE, 3)
    lm = b.build()
    raw = backend.solve(lm, FAST)
    assert raw.objective == pytest.approx(-1.0)
    assert raw.duals is not None
    np.testing.assert_allclose(raw.duals, [1.0, 0.5, -1.0], atol=1e-8)
    assert dual_sign_violations(lm, raw.duals) == []
    assert dual_objective(lm, raw.duals) == pytest.approx(-1.0)


def test_mip_optimum(backend: SolverBackend) -> None:
    raw = backend.solve(knapsack(), SolverSettings(mip_gap=0))
    assert raw.status is SolveStatus.OPTIMAL
    assert raw.objective == pytest.approx(23.0)
    assert raw.x is not None
    np.testing.assert_allclose(raw.x, [1, 1, 0, 0], atol=1e-6)
    assert raw.bound == pytest.approx(23.0, rel=1e-6)
    assert raw.duals is None  # no duals for MIPs


def test_warm_start_is_accepted(backend: SolverBackend) -> None:
    raw = backend.solve(knapsack(), FAST, warm_start=np.array([0.0, 1.0, 0.0, 1.0]))
    assert raw.objective == pytest.approx(23.0)


def test_infeasible_model(backend: SolverBackend) -> None:
    b = ModelBuilder("infeasible")
    x = b.add_vars("x", 1)
    b.add_constrs("lo", [(x, [[1.0]])], Sense.GE, 2)
    b.add_constrs("hi", [(x, [[1.0]])], Sense.LE, 1)
    raw = backend.solve(b.build(), FAST)
    assert raw.status in {SolveStatus.INFEASIBLE, SolveStatus.INF_OR_UNBD}
    assert not raw.has_solution and raw.objective is None


def test_unbounded_model(backend: SolverBackend) -> None:
    b = ModelBuilder("unbounded", ObjSense.MAXIMIZE)
    x = b.add_vars("x", 2, obj=1.0)
    b.add_constrs("c", [(x, [[1, -1]])], Sense.LE, 1)
    raw = backend.solve(b.build(), FAST)
    assert raw.status in {SolveStatus.UNBOUNDED, SolveStatus.INF_OR_UNBD}


def test_lazy_constraints_reach_true_optimum(backend: SolverBackend) -> None:
    # max x + y over integers in [0, 10]^2; the oracle enforces x + y <= 7 lazily.
    b = ModelBuilder("lazy", ObjSense.MAXIMIZE)
    xy = b.add_vars("xy", 2, ub=10, vtype=VarType.INTEGER, obj=1.0)
    b.add_constrs("dummy", [(xy, [[1, 0]])], Sense.LE, 10)
    cut = Cut(np.array([0, 1]), np.array([1.0, 1.0]), Sense.LE, 7.0)

    def oracle(x: np.ndarray) -> list[Cut]:
        return [cut] if cut.violation(x) > 1e-6 else []

    raw = backend.solve(b.build(), FAST, lazy=oracle)
    assert raw.status is SolveStatus.OPTIMAL
    assert raw.objective == pytest.approx(7.0)
    assert raw.extra["lazy_cuts"] >= 1


def test_export_writes_model_file(backend: SolverBackend, tmp_path: Path) -> None:
    target = tmp_path / "wyndor.lp"
    backend.solve(wyndor(), SolverSettings(export=target))
    assert target.exists() and "cap" in target.read_text()


def test_large_model_solves_or_reports_licence_limit(backend: SolverBackend) -> None:
    """Never leak a raw solver error: either solve, or raise LicenseLimitError."""
    n = 2500
    b = ModelBuilder("big")
    x = b.add_vars("x", n, obj=np.linspace(1, 2, n))
    b.add_constrs("total", [(x, np.ones((1, n)))], Sense.GE, 1)
    try:
        raw = backend.solve(b.build(), FAST)
    except LicenseLimitError as exc:
        assert "licence" in str(exc)  # noqa: PT017 - either outcome is acceptable
    else:
        assert raw.objective == pytest.approx(1.0)


# ------------------------------------------------------------- Gurobi-only
def test_gurobi_records_mip_progress(gurobi: SolverBackend) -> None:
    raw = gurobi.solve(knapsack(), FAST)
    assert gurobi.supports(Capability.PROGRESS)
    assert raw.progress, "callback should record at least the final incumbent"
    assert raw.progress[-1].incumbent == pytest.approx(23.0)


def test_gurobi_iis_names_the_conflict(gurobi: SolverBackend) -> None:
    b = ModelBuilder("infeasible")
    x = b.add_vars("x", 2)
    b.add_constrs("lo", [(x, [[1, 0]])], Sense.GE, 2)
    b.add_constrs("hi", [(x, [[1, 0]])], Sense.LE, 1)
    b.add_constrs("other", [(x, [[0, 1]])], Sense.LE, 5)
    iis = gurobi.compute_iis(b.build(), FAST)
    assert set(iis) == {"lo[0]", "hi[0]"}


def test_gurobi_iis_is_empty_for_feasible_model(gurobi: SolverBackend) -> None:
    assert gurobi.compute_iis(wyndor(), FAST) == []


def test_iis_unsupported_backend_raises() -> None:
    from linear_opt.core.backends import is_available
    from linear_opt.core.config import BackendName

    if not is_available(BackendName.HIGHS):
        pytest.skip("highspy not installed")
    from linear_opt.solvers.highs import HighsBackend

    with pytest.raises(NotImplementedError):
        HighsBackend().compute_iis(wyndor(), FAST)


def test_highs_records_mip_progress() -> None:
    from linear_opt.core.backends import is_available
    from linear_opt.core.config import BackendName

    if not is_available(BackendName.HIGHS):
        pytest.skip("highspy not installed")
    from linear_opt.solvers.highs import HighsBackend

    raw = HighsBackend().solve(knapsack(), FAST)
    assert raw.progress
    assert raw.progress[-1].incumbent == pytest.approx(23.0)


def test_native_parameters_are_applied_and_validated(backend: SolverBackend) -> None:
    gurobi = backend.name.value == "gurobi"
    good = {"gurobi": {"Cuts": 0, "Presolve": 0}} if gurobi else {"highs": {"presolve": "off"}}
    raw = backend.solve(knapsack(), SolverSettings(**good))
    assert raw.objective == pytest.approx(23.0)
    bad = {"gurobi": {"NoSuchParam": 1}} if gurobi else {"highs": {"no_such_option": 1}}
    with pytest.raises(Exception, match=r"(?i)param|option"):
        backend.solve(knapsack(), SolverSettings(**bad))


def test_relaxation_drops_integrality_only() -> None:
    lm = knapsack()
    lp = lm.relaxed()
    assert lm.is_mip and not lp.is_mip
    assert lp.num_vars == lm.num_vars
    assert (lp.ub == lm.ub).all()
    assert lp.name == "knapsack_lp"
