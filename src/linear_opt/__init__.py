"""linear_opt: production-grade LP/MILP modelling with Gurobi and a HiGHS fallback."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("linear-opt")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
