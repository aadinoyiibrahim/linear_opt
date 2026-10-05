"""Problem-independent MIP figures: solve progress and formulation comparison."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

from linear_opt.core.result import ProgressPoint
from linear_opt.core.study import VariantResult
from linear_opt.viz.theme import palette, template

INCUMBENT, BOUND = 0, 1  # categorical slots


def _x_axis(points: Sequence[ProgressPoint]) -> tuple[list[float], str]:
    times = [p.time for p in points]
    if len(points) > 1 and max(times) - min(times) < 1e-3:
        return list(map(float, range(len(points)))), "event #"
    return times, "time (s)"


def mip_progress(
    progress: Sequence[ProgressPoint],
    *,
    reference: float | None = None,
    title: str = "Branch-and-bound progress",
    dark: bool = False,
) -> go.Figure:
    """Incumbent and best bound over time; the gap is the space between them."""
    p = palette(dark)
    fig = go.Figure()
    x, x_title = _x_axis(progress)
    series = (
        ("incumbent (best solution)", [pt.incumbent for pt in progress], INCUMBENT),
        ("bound (best possible)", [pt.bound for pt in progress], BOUND),
    )
    for name, y, slot in series:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                mode="lines+markers",
                line={"shape": "hv", "width": 2, "color": p.categorical[slot]},
                marker={"size": 8, "line": {"width": 2, "color": p.surface}},
                connectgaps=False,
                hovertemplate="<b>%{y:,.2f}</b><br>" + name + "<br>%{x}<extra></extra>",
            )
        )
    if reference is not None:
        fig.add_hline(
            y=reference,
            line={"dash": "dot", "width": 1.5, "color": p.muted},
            annotation_text=f"known optimum {reference:,.2f}",
            annotation_font={"color": p.ink_secondary, "size": 12},
            annotation_position="top left",
        )
    # Solvers start from trivial bounds (e.g. 0) that would flatten the
    # interesting region; zoom the y-axis onto it (pan/zoom still shows all).
    finals = (
        [v for v in (progress[-1].incumbent, progress[-1].bound) if v is not None]
        if progress
        else []
    )
    values = finals + [pt.incumbent for pt in progress if pt.incumbent is not None]
    if reference is not None:
        values.append(reference)
    if values and max(values) - min(values) <= 1e-9 * max(1.0, abs(max(values))):
        # Only one objective value (e.g. row generation shows no incumbents):
        # fall back to the bound history so its climb remains visible.
        values += [pt.bound for pt in progress if pt.bound is not None]
    yaxis: dict[str, object] = {"title": "objective"}
    if values:
        lo, hi = min(values), max(values)
        span = max(hi - lo, 1e-9 * max(1.0, abs(hi)))
        yaxis["range"] = [lo - 0.6 * span, hi + 0.15 * span]
    fig.update_layout(
        template=template(dark),
        title=title,
        xaxis_title=x_title,
        yaxis=yaxis,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 1, "xanchor": "right"},
        height=420,
    )
    return fig


def variant_root_gaps(results: Sequence[VariantResult], *, dark: bool = False) -> go.Figure:
    """Root gap (LP relaxation vs MIP optimum) for each formulation."""
    p = palette(dark)
    names = [r.name for r in results]
    gaps = np.array([np.nan if r.root_gap is None else 100 * r.root_gap for r in results])
    lp = [r.lp_bound for r in results]
    fig = go.Figure(
        go.Bar(
            x=gaps,
            y=names,
            orientation="h",
            marker={"color": p.categorical[0], "line": {"width": 0}, "cornerradius": 4},
            text=[f"{g:.2f}%" for g in gaps],
            textposition="outside",
            textfont={"color": p.ink_secondary},
            customdata=lp,
            hovertemplate="<b>%{x:.3f}%</b> root gap<br>LP bound %{customdata:,.2f}"
            "<br>%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        template=template(dark),
        title="LP-relaxation gap by formulation (smaller = tighter)",
        xaxis={"title": "root gap (%)", "rangemode": "tozero"},
        yaxis={"showgrid": False, "autorange": "reversed"},
        height=max(240, 60 * len(results) + 120),
        bargap=0.35,
    )
    return fig


def variant_effort(results: Sequence[VariantResult], *, dark: bool = False) -> list[go.Figure]:
    """Runtime and branch-and-bound nodes by formulation (two charts, one axis each)."""
    p = palette(dark)
    names = [r.name for r in results]
    figs = []
    for metric, label, values in (
        ("runtime", "runtime (s)", [r.result.raw.runtime for r in results]),
        ("nodes", "branch-and-bound nodes", [r.result.raw.nodes for r in results]),
    ):
        fig = go.Figure(
            go.Bar(
                x=values,
                y=names,
                orientation="h",
                marker={"color": p.categorical[0], "line": {"width": 0}, "cornerradius": 4},
                hovertemplate="<b>%{x:,}</b> " + label + "<br>%{y}<extra></extra>",
            )
        )
        fig.update_layout(
            template=template(dark),
            title=f"Solver effort: {metric}",
            xaxis={"title": label, "rangemode": "tozero"},
            yaxis={"showgrid": False, "autorange": "reversed"},
            height=max(220, 60 * len(results) + 100),
            bargap=0.35,
        )
        figs.append(fig)
    return figs


def variant_progress(results: Sequence[VariantResult], *, dark: bool = False) -> go.Figure:
    """Relative gap over time for up to three formulations (one hue each)."""
    p = palette(dark)
    fig = go.Figure()
    for slot, r in enumerate(results[:3]):
        pairs = [
            (pt.time, pt.incumbent, pt.bound)
            for pt in r.result.raw.progress
            if pt.incumbent is not None and pt.bound is not None
        ]
        if not pairs:
            continue
        gap = [100 * abs(inc - bnd) / max(abs(inc), 1e-10) for _, inc, bnd in pairs]
        fig.add_trace(
            go.Scatter(
                x=[t for t, _, _ in pairs],
                y=gap,
                name=r.name,
                mode="lines+markers",
                line={"shape": "hv", "width": 2, "color": p.categorical[slot]},
                marker={"size": 8, "line": {"width": 2, "color": p.surface}},
                hovertemplate="<b>%{y:.3f}%</b> gap<br>" + r.name + " · %{x:.3f}s<extra></extra>",
            )
        )
    fig.update_layout(
        template=template(dark),
        title="Optimality gap over time",
        xaxis_title="time (s)",
        yaxis={"title": "gap (%)", "rangemode": "tozero"},
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 1, "xanchor": "right"},
        height=400,
    )
    return fig
