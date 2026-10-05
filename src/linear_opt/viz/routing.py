"""Figures for the TSP and the CVRP."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

from linear_opt.core.ir import FloatArray
from linear_opt.problems.cvrp import CVRPInstance, CVRPSolution
from linear_opt.problems.tsp import TSPInstance, TSPSolution
from linear_opt.viz.mip import mip_progress
from linear_opt.viz.theme import palette, template

NODE, DEPOT = 1, 0  # categorical slots


def _coords(xy: FloatArray | None, n: int) -> FloatArray:
    if xy is not None:
        return xy
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)  # no coordinates: draw on a circle
    return np.column_stack([np.cos(angle), np.sin(angle)])


def _trace(geographic: bool) -> Any:
    """Trace class: map projection for lat/lon, plain axes otherwise."""
    return go.Scattergeo if geographic else go.Scatter


def _xy(points: FloatArray, geographic: bool) -> dict[str, FloatArray]:
    if geographic:
        return {"lat": points[:, 0], "lon": points[:, 1]}
    return {"x": points[:, 0], "y": points[:, 1]}


def _frame(fig: go.Figure, geographic: bool, height: int = 620) -> go.Figure:
    if geographic:
        fig.update_geos(fitbounds="locations", projection_type="mercator", resolution=50)
    else:
        fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, constrain="domain")
        fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x")
    fig.update_layout(height=height)
    return fig


def _segments(points: FloatArray, edges: Sequence[tuple[int, int]]) -> FloatArray:
    """Edges as one polyline with NaN separators (one trace, fast to draw)."""
    out = np.full((3 * len(edges), 2), np.nan)
    for k, (a, b) in enumerate(edges):
        out[3 * k], out[3 * k + 1] = points[a], points[b]
    return out


def tour_map(inst: TSPInstance, sol: TSPSolution, *, dark: bool = False) -> go.Figure:
    """The optimal tour; hover a city for its position in the tour."""
    p = palette(dark)
    pts = _coords(inst.coords, inst.n)
    closed = np.array([*sol.tour, sol.tour[0]])
    trace = _trace(inst.geographic)
    position = {v: k for k, v in enumerate(sol.tour)}
    fig = go.Figure(
        [
            trace(
                **_xy(pts[closed], inst.geographic),
                mode="lines",
                line={"width": 2, "color": p.categorical[0]},
                hoverinfo="skip",
                name="tour",
                showlegend=False,
            ),
            trace(
                **_xy(pts, inst.geographic),
                mode="markers",
                marker={
                    "size": 9,
                    "color": p.categorical[NODE],
                    "line": {"width": 2, "color": p.surface},
                },
                text=[
                    f"<b>{name}</b><br>stop {position[v] + 1} of {inst.n}"
                    for v, name in enumerate(inst.names)
                ],
                hoverinfo="text",
                name="city",
                showlegend=False,
            ),
        ]
    )
    title = f"Optimal tour · length {sol.length:,.0f} {inst.unit}"
    if inst.reference is not None:
        title += f" (published optimum {inst.reference:,.0f})"
    fig.update_layout(template=template(dark), title=title)
    return _frame(fig, inst.geographic)


def subtour_history(inst: TSPInstance, sol: TSPSolution, *, dark: bool = False) -> go.Figure:
    """Replay the integer candidates the separator rejected, ending with the tour.

    Each frame shows one candidate of the lazy-constraint loop; its subtours are
    what the next cuts eliminate. Use the slider or press play.
    """
    p = palette(dark)
    pts = _coords(inst.coords, inst.n)
    frames_src = [f for f in sol.history if f.n_subtours > 1][:60]
    final = (tuple(sol.edges()), 1)
    steps = [(f.edges, f.n_subtours) for f in frames_src] + [final]
    trace = _trace(inst.geographic)

    def edge_trace(edges: Sequence[tuple[int, int]]) -> go.Scatter | go.Scattergeo:
        return trace(
            **_xy(_segments(pts, edges), inst.geographic),
            mode="lines",
            line={"width": 2, "color": p.categorical[0]},
            hoverinfo="skip",
            showlegend=False,
        )

    cities = trace(
        **_xy(pts, inst.geographic),
        mode="markers",
        marker={"size": 8, "color": p.categorical[NODE], "line": {"width": 2, "color": p.surface}},
        text=list(inst.names),
        hoverinfo="text",
        showlegend=False,
    )

    def label(k: int, n_sub: int) -> str:
        if k == len(steps) - 1:
            return "optimal tour"
        return f"candidate {k + 1}: {n_sub} subtours"

    fig = go.Figure(
        data=[edge_trace(steps[0][0]), cities],
        frames=[
            go.Frame(
                data=[edge_trace(edges)],
                traces=[0],
                name=str(k),
                layout={"title": {"text": label(k, n)}},
            )
            for k, (edges, n) in enumerate(steps)
        ],
    )
    fig.update_layout(
        template=template(dark),
        title=label(0, steps[0][1]),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": -0.06,
                "xanchor": "left",
                "buttons": [
                    {
                        "label": "▶ play",
                        "method": "animate",
                        "args": [
                            None,
                            {"frame": {"duration": 700, "redraw": True}, "fromcurrent": True},
                        ],
                    }
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.12,
                "len": 0.88,
                "y": -0.04,
                "currentvalue": {"visible": False},
                "steps": [
                    {
                        "label": str(k + 1),
                        "method": "animate",
                        "args": [
                            [str(k)],
                            {"mode": "immediate", "frame": {"duration": 0, "redraw": True}},
                        ],
                    }
                    for k in range(len(steps))
                ],
            }
        ],
    )
    return _frame(fig, inst.geographic, height=660)


def route_map(inst: CVRPInstance, sol: CVRPSolution, *, dark: bool = False) -> go.Figure:
    """Vehicle routes from the depot; each route is labelled with its load."""
    p = palette(dark)
    pts = _coords(inst.coords, inst.n)
    trace = _trace(inst.geographic)
    fig = go.Figure()
    for k, (route, load, cost) in enumerate(zip(sol.routes, sol.loads, sol.costs, strict=True)):
        path = np.array([0, *route, 0])
        fig.add_trace(
            trace(
                **_xy(pts[path], inst.geographic),
                mode="lines",
                line={"width": 2, "color": p.ink_secondary},
                opacity=0.8,
                name=f"route {k + 1}",
                text=f"<b>route {k + 1}</b><br>load {load:g} / {inst.capacity:g}<br>cost {cost:,.0f}",
                hoverinfo="text",
                showlegend=False,
            )
        )
        mid = pts[list(route)].mean(axis=0)  # inside the route's petal, away from markers
        fig.add_trace(
            trace(
                **_xy(mid[None, :], inst.geographic),
                mode="text",
                text=[f"R{k + 1} · {load:g}/{inst.capacity:g}"],
                textposition="middle center",
                textfont={"color": p.ink_secondary, "size": 11},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        trace(
            **_xy(pts[1:], inst.geographic),
            mode="markers",
            name="customer (size = demand)",
            marker={
                "size": 6 + 14 * np.sqrt(inst.demand[1:] / max(inst.demand[1:].max(), 1e-12)),
                "color": p.categorical[NODE],
                "line": {"width": 2, "color": p.surface},
            },
            text=[f"<b>{inst.names[v]}</b><br>demand {inst.demand[v]:g}" for v in range(1, inst.n)],
            hoverinfo="text",
        )
    )
    fig.add_trace(
        trace(
            **_xy(pts[:1], inst.geographic),
            mode="markers",
            name="depot",
            marker={
                "size": 18,
                "symbol": "square",
                "color": p.categorical[DEPOT],
                "line": {"width": 2, "color": p.surface},
            },
            text=[f"<b>{inst.names[0]}</b> (depot)"],
            hoverinfo="text",
        )
    )
    title = f"{len(sol.routes)} routes · total cost {sol.total_cost:,.0f} {inst.unit}"
    if inst.reference is not None:
        title += f" (published optimum {inst.reference:,.0f})"
    fig.update_layout(
        template=template(dark), title=title, legend={"orientation": "h", "y": -0.02, "x": 0}
    )
    return _frame(fig, inst.geographic)


def route_loads(inst: CVRPInstance, sol: CVRPSolution, *, dark: bool = False) -> go.Figure:
    """Load of each route against the vehicle capacity."""
    p = palette(dark)
    names = [f"R{k + 1}" for k in range(len(sol.routes))]
    fig = go.Figure(
        go.Bar(
            x=list(sol.loads),
            y=names,
            orientation="h",
            marker={"color": p.categorical[0], "line": {"width": 0}, "cornerradius": 4},
            customdata=np.column_stack([[len(r) for r in sol.routes], sol.costs]),
            hovertemplate="<b>%{x:g}</b> loaded<br>%{customdata[0]} stops · cost %{customdata[1]:,.0f}"
            "<br>%{y}<extra></extra>",
        )
    )
    fig.add_vline(
        x=inst.capacity,
        line={"dash": "dot", "width": 1.5, "color": p.muted},
        annotation_text=f"capacity {inst.capacity:g}",
        annotation_font={"color": p.ink_secondary, "size": 12},
    )
    fig.update_layout(
        template=template(dark),
        title=f"Vehicle loads · fleet {len(sol.routes)} of {inst.vehicles}",
        xaxis={"title": "load", "range": [0, inst.capacity * 1.12]},
        yaxis={"showgrid": False, "autorange": "reversed"},
        height=max(240, 32 * len(names) + 120),
        bargap=0.3,
    )
    return fig


def tsp_figures(inst: TSPInstance, sol: TSPSolution, *, dark: bool = False) -> list[go.Figure]:
    """Every TSP figure, in reading order."""
    figs = [tour_map(inst, sol, dark=dark)]
    if any(f.n_subtours > 1 for f in sol.history):
        figs.append(subtour_history(inst, sol, dark=dark))
    if sol.progress:
        figs.append(mip_progress(sol.progress, reference=inst.reference, dark=dark))
    return figs


def cvrp_figures(inst: CVRPInstance, sol: CVRPSolution, *, dark: bool = False) -> list[go.Figure]:
    """Every CVRP figure, in reading order."""
    figs = [route_map(inst, sol, dark=dark), route_loads(inst, sol, dark=dark)]
    if sol.progress:
        figs.append(mip_progress(sol.progress, reference=inst.reference, dark=dark))
    return figs
