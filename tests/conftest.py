from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

from linear_opt.core.backends import is_available
from linear_opt.core.config import BackendName
from linear_opt.solvers import SolverBackend, create_backend

# CI runs property tests derandomised (reproducible failures); locally they explore.
settings.register_profile(
    "ci", derandomize=True, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile("dev", deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
INSTALLED = [b for b in (BackendName.GUROBI, BackendName.HIGHS) if is_available(b)]

WriteToml = Callable[..., Path]


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Never touch the real download cache from tests."""
    monkeypatch.setenv("LINEAR_OPT_DATA_DIR", str(tmp_path_factory.mktemp("data")))


@pytest.fixture(params=INSTALLED, ids=str)
def backend(request: pytest.FixtureRequest) -> SolverBackend:
    """Every installed solver backend."""
    return create_backend(request.param)


@pytest.fixture
def gurobi() -> SolverBackend:
    if not is_available(BackendName.GUROBI):
        pytest.skip("gurobipy not installed")
    return create_backend(BackendName.GUROBI)


@pytest.fixture
def write_toml(tmp_path: Path) -> WriteToml:
    """Write a TOML string to a temporary file and return its path."""

    def _write(text: str, name: str = "run.toml") -> Path:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    return _write


def satisfies(lm, x, tol: float = 1e-6) -> list[str]:  # type: ignore[no-untyped-def]
    """Rows and bounds of ``lm`` violated by ``x`` (solver-independent check)."""
    import numpy as np

    issues = []
    if np.any(x < lm.lb - tol) or np.any(x > lm.ub + tol):
        issues.append("bounds")
    ax = lm.A @ x
    names = lm.constr_names()
    for k, (sense, lhs, rhs) in enumerate(zip(lm.senses, ax, lm.rhs, strict=True)):
        if (sense == "<" and lhs > rhs + tol) or (sense == ">" and lhs < rhs - tol):
            issues.append(names[k])
        if sense == "=" and abs(lhs - rhs) > tol:
            issues.append(names[k])
    ints = lm.integer_mask
    if np.any(np.abs(x[ints] - np.round(x[ints])) > tol):
        issues.append("integrality")
    return issues
