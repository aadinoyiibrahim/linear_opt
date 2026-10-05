"""Discovery of installed solver backends.

The concrete ``SolverBackend`` protocol and its Gurobi/HiGHS implementations
arrive in Phase 1; this module only answers "what can I run here?".
"""

from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError, version

from linear_opt.core.config import BackendName

_MODULES: dict[BackendName, tuple[str, str]] = {
    # backend -> (import name, distribution name)
    BackendName.GUROBI: ("gurobipy", "gurobipy"),
    BackendName.HIGHS: ("highspy", "highspy"),
}

#: Order in which ``auto`` tries backends.
PREFERENCE: tuple[BackendName, ...] = (BackendName.GUROBI, BackendName.HIGHS)


class BackendUnavailableError(RuntimeError):
    """Raised when the requested solver backend is not installed."""


def is_available(backend: BackendName) -> bool:
    """Return whether the Python package for ``backend`` is importable."""
    if backend is BackendName.AUTO:
        return any(is_available(b) for b in PREFERENCE)
    return importlib.util.find_spec(_MODULES[backend][0]) is not None


def available_backends() -> dict[BackendName, str]:
    """Map each installed backend to its package version."""
    found: dict[BackendName, str] = {}
    for backend in PREFERENCE:
        if is_available(backend):
            try:
                found[backend] = version(_MODULES[backend][1])
            except PackageNotFoundError:  # pragma: no cover - importable but not a dist
                found[backend] = "unknown"
    return found


def resolve_backend(requested: BackendName) -> BackendName:
    """Turn a requested backend (possibly ``auto``) into a concrete installed one.

    Raises:
        BackendUnavailableError: If nothing suitable is installed.
    """
    if requested is not BackendName.AUTO:
        if not is_available(requested):
            raise BackendUnavailableError(
                f"backend '{requested}' is not installed; try `uv sync --extra {requested}`"
            )
        return requested
    for backend in PREFERENCE:
        if is_available(backend):
            return backend
    raise BackendUnavailableError(
        "no solver backend installed; try `uv sync --extra highs` or `--extra gurobi`"
    )
