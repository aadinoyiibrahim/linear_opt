from __future__ import annotations

import pytest

from linear_opt.core import backends
from linear_opt.core.backends import (
    BackendUnavailableError,
    available_backends,
    resolve_backend,
)
from linear_opt.core.config import BackendName


def _fake_installed(monkeypatch: pytest.MonkeyPatch, installed: set[BackendName]) -> None:
    def fake(backend: BackendName) -> bool:
        if backend is BackendName.AUTO:
            return bool(installed)
        return backend in installed

    monkeypatch.setattr(backends, "is_available", fake)


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        ({BackendName.GUROBI, BackendName.HIGHS}, BackendName.GUROBI),  # Gurobi preferred
        ({BackendName.HIGHS}, BackendName.HIGHS),  # graceful fallback
        ({BackendName.GUROBI}, BackendName.GUROBI),
    ],
)
def test_auto_prefers_gurobi_then_highs(
    monkeypatch: pytest.MonkeyPatch, installed: set[BackendName], expected: BackendName
) -> None:
    _fake_installed(monkeypatch, installed)
    assert resolve_backend(BackendName.AUTO) is expected


def test_auto_without_any_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_installed(monkeypatch, set())
    with pytest.raises(BackendUnavailableError, match="no solver backend"):
        resolve_backend(BackendName.AUTO)


def test_explicit_missing_backend_raises_with_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_installed(monkeypatch, {BackendName.HIGHS})
    with pytest.raises(BackendUnavailableError, match="--extra gurobi"):
        resolve_backend(BackendName.GUROBI)


def test_available_backends_reports_real_installation() -> None:
    found = available_backends()
    assert set(found) <= {BackendName.GUROBI, BackendName.HIGHS}
    assert all(isinstance(v, str) and v for v in found.values())
