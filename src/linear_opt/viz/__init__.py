"""Plotly figures: pure functions mapping (instance, solution) to figures.

Requires the ``viz`` extra: ``uv sync --extra viz``.
"""

from __future__ import annotations

import importlib
from typing import Any

#: problem kind -> (module in this package, figure function)
FIGURES: dict[str, tuple[str, str]] = {
    "transport": ("transport", "all_figures"),
    "facility_location": ("facility", "all_figures"),
    "tsp": ("routing", "tsp_figures"),
    "cvrp": ("routing", "cvrp_figures"),
    "job_shop": ("scheduling", "all_figures"),
}


def figures_for(kind: str, data: Any, solution: Any, *, dark: bool = False) -> list[Any]:
    """All figures for a solved instance of problem ``kind`` (empty if unknown)."""
    if solution is None or kind not in FIGURES:
        return []
    module, function = FIGURES[kind]
    make = getattr(importlib.import_module(f"linear_opt.viz.{module}"), function)
    return list(make(data, solution, dark=dark))
