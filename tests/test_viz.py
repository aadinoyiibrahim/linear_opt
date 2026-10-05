from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("plotly")

import plotly.io as pio

from linear_opt.core.config import SolverSettings
from linear_opt.core.result import ProgressPoint
from linear_opt.core.study import run_variants
from linear_opt.problems.facility import FacilityInstance, FacilityModel
from linear_opt.problems.transport import TransportInstance, TransportModel
from linear_opt.viz import facility as fviz
from linear_opt.viz import mip
from linear_opt.viz import transport as tviz
from linear_opt.viz.report import fmt, model_subtitle, run_kpis, write_report
from linear_opt.viz.theme import TEMPLATE_DARK, TEMPLATE_LIGHT

from .test_transport import CITIES


@pytest.fixture(scope="module")
def solved() -> Any:
    model = TransportModel(TransportInstance.random(4, 9, seed=5))
    result = model.solve()
    assert result.solution is not None
    return model, result


@pytest.fixture(scope="module")
def facility() -> Any:
    model = FacilityModel(FacilityInstance.random(6, 15, seed=2, capacity_ratio=1.5))
    result = model.solve()
    assert result.solution is not None
    return model, result


# ------------------------------------------------------------------ transport
def test_transport_figures(solved: Any) -> None:
    model, result = solved
    figs = tviz.all_figures(model.data, result.solution)
    assert len(figs) == 4
    flow_map = figs[0]
    assert len(flow_map.data) == len(result.solution.routes()) + 2  # routes + depots + customers


def test_dark_mode_uses_dark_template(solved: Any) -> None:
    model, result = solved
    light = tviz.flow_matrix(model.data, result.solution)
    dark = tviz.flow_matrix(model.data, result.solution, dark=True)
    assert light.layout.template.layout.paper_bgcolor != dark.layout.template.layout.paper_bgcolor
    assert {TEMPLATE_LIGHT, TEMPLATE_DARK} <= set(pio.templates)


def test_geographic_instance_uses_map_projection() -> None:
    inst = TransportInstance.from_cities(CITIES, n_sources=2)
    sol = TransportModel(inst).solve().solution
    assert sol is not None
    assert all(t.type == "scattergeo" for t in tviz.flow_map(inst, sol).data)


def test_shadow_prices_need_duals(solved: Any) -> None:
    model, result = solved
    with pytest.raises(ValueError, match="duals"):
        tviz.shadow_prices(model.data, replace(result.solution, demand_duals=None))


# ------------------------------------------------------------------- facility
def test_facility_figures(facility: Any) -> None:
    model, result = facility
    sol = result.solution
    figs = fviz.all_figures(model.data, sol)
    assert len(figs) == 5  # map, cost, utilisation, progress, assignments
    n_open = int(sol.open.sum())
    assert len(figs[1].data) == 2 and len(figs[1].data[0].y) == n_open  # stacked cost bars
    assert len(figs[2].data[0].y) == n_open


def test_facility_map_requires_coordinates(facility: Any) -> None:
    model, result = facility
    bare = replace(model.data, facility_xy=None, customer_xy=None)
    with pytest.raises(ValueError, match="coordinates"):
        fviz.facility_map(bare, result.solution)
    assert len(fviz.all_figures(bare, result.solution)) == 4


def test_mip_progress_with_reference_zooms_y_axis() -> None:
    pts = [
        ProgressPoint(0.0, None, 0.0),
        ProgressPoint(0.1, 120.0, 90.0),
        ProgressPoint(0.2, 100.0, 100.0),
    ]
    fig = mip.mip_progress(pts, reference=100.0)
    lo, hi = fig.layout.yaxis.range
    assert 0.0 < lo < 90.0 < 120.0 < hi  # trivial bound 0 is out of the default view
    assert fig.layout.shapes  # reference line


def test_mip_progress_uses_event_index_for_instant_solves() -> None:
    pts = [ProgressPoint(0.0, 5.0, 1.0), ProgressPoint(0.0, 4.0, 4.0)]
    assert mip.mip_progress(pts).layout.xaxis.title.text == "event #"


def test_variant_figures(facility: Any) -> None:
    model, _ = facility
    results = run_variants(model, SolverSettings(time_limit=10))
    assert len(mip.variant_root_gaps(results).data[0].x) == 3
    assert len(mip.variant_effort(results)) == 2
    assert 1 <= len(mip.variant_progress(results).data) <= 3


# --------------------------------------------------------------------- report
def test_report_contains_kpis_and_figures(solved: Any, tmp_path: Path) -> None:
    model, result = solved
    summary = result.summary()
    figs = tviz.all_figures(model.data, result.solution)
    path = write_report(
        tmp_path / "r.html",
        "demo <run>",
        figs,
        kpis=run_kpis(summary),
        subtitle=model_subtitle(summary),
    )
    text = path.read_text(encoding="utf-8")
    assert "demo &lt;run&gt;" in text  # title is escaped
    assert "optimal" in text and text.count("<section>") == 4
    assert "variables: 36" in text


def test_report_formatting_and_reference_tiles() -> None:
    assert fmt(None) == "—"
    assert fmt(True) == "yes"
    assert fmt(1234.5) == "1,234.50"
    assert fmt(2e-5) == "2.00e-05"
    assert fmt(3_000_000.4) == "3,000,000"
    tiles = run_kpis({"status": "optimal", "reference": 10.0, "reference_gap": 0.0})
    assert tiles["Known optimum"] == 10.0
    assert "Deviation" in tiles


# -------------------------------------------------------------------- routing
def test_tsp_figures_include_subtour_replay() -> None:
    from linear_opt.problems.tsp import TSPInstance, TSPModel
    from linear_opt.viz import routing

    model = TSPModel(TSPInstance.random(40, seed=9), {"root_cuts": False, "warm_start": "none"})
    sol = model.solve(
        SolverSettings(), backend="highs" if "highs" in _installed() else None
    ).solution
    assert sol is not None
    figs = routing.tsp_figures(model.data, sol)
    assert figs[0].data[0].mode == "lines"
    if any(f.n_subtours > 1 for f in sol.history):
        replay = figs[1]
        assert len(replay.frames) >= 2
        assert replay.frames[-1].layout.title.text == "optimal tour"


def test_cvrp_figures() -> None:
    from linear_opt.problems.cvrp import CVRPInstance, CVRPModel
    from linear_opt.viz import routing

    model = CVRPModel(CVRPInstance.random(8, seed=1, capacity=50))
    sol = model.solve().solution
    assert sol is not None
    figs = routing.cvrp_figures(model.data, sol)
    assert len(figs) >= 2
    assert len(figs[1].data[0].x) == len(sol.routes)  # one load bar per route


def test_report_does_not_autoplay(tmp_path: Path) -> None:
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(x=[0], y=[0]), frames=[go.Frame(data=[go.Scatter(x=[1], y=[1])])])
    text = write_report(tmp_path / "a.html", "anim", [fig], kpis={}).read_text()
    assert "Plotly.animate" not in text


def _installed() -> set[str]:
    from .conftest import INSTALLED

    return {b.value for b in INSTALLED}


# ----------------------------------------------------------------- scheduling
def test_gantt_figures_mark_the_critical_path() -> None:
    from linear_opt.problems.jobshop import JobShopInstance, JobShopModel
    from linear_opt.viz import scheduling

    inst = JobShopInstance.random(4, 3, seed=1)
    sol = JobShopModel(inst).solve().solution
    assert sol is not None
    figs = scheduling.all_figures(inst, sol)
    assert len(figs) >= 3
    bars = [t for t in figs[0].data if t.base is not None]
    assert len(bars) == inst.n_jobs
    outlined = sum(sum(w > 1 for w in t.marker.line.width) for t in bars)
    assert outlined == len(sol.critical)
    swatches = [t for t in figs[0].data if t.base is None and t.name.startswith("job")]
    assert all(t.marker.line.width == 0 for t in swatches)  # legend has no outline leak
