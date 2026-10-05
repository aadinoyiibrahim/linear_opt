"""Figures for the transportation problem."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from linear_opt.problems.transport import TransportInstance, TransportSolution
from linear_opt.viz.theme import palette, sequential_scale, template

DEPOT, CUSTOMER = 0, 1  # categorical slots


def _xy(points: np.ndarray, geographic: bool) -> dict[str, np.ndarray]:
    return (
        {"lat": points[:, 0], "lon": points[:, 1]}
        if geographic
        else {"x": points[:, 0], "y": points[:, 1]}
    )


def flow_map(inst: TransportInstance, sol: TransportSolution, *, dark: bool = False) -> go.Figure:
    """Depots, customers and the optimal shipments; line width proportional to flow."""
    p = palette(dark)
    trace = go.Scattergeo if inst.geographic else go.Scatter
    fig = go.Figure()

    routes = sol.routes()
    peak = max((f for _, _, f in routes), default=1.0)
    for i, j, f in routes:
        ends = np.vstack([inst.source_xy[i], inst.sink_xy[j]])
        fig.add_trace(
            trace(
                **_xy(ends, inst.geographic),
                mode="lines",
                line={"width": 1.0 + 6.0 * f / peak, "color": p.ink_secondary},
                opacity=0.55,
                hoverinfo="text",
                text=f"<b>{f:,.1f}</b> units<br>{inst.sources[i]} → {inst.sinks[j]}"
                f"<br>{inst.cost[i, j]:,.1f} {inst.cost_unit} per unit",
                showlegend=False,
            )
        )

    used = sol.flow.sum(axis=1)
    for slot, label, xy, mass, hover in (
        (
            DEPOT,
            "Depot (size = capacity)",
            inst.source_xy,
            inst.supply,
            [
                f"<b>{n}</b><br>ships {u:,.1f} of {s:,.1f}"
                for n, u, s in zip(inst.sources, used, inst.supply, strict=True)
            ],
        ),
        (
            CUSTOMER,
            "Customer (size = demand)",
            inst.sink_xy,
            inst.demand,
            [
                f"<b>{n}</b><br>demand {d:,.1f}"
                for n, d in zip(inst.sinks, inst.demand, strict=True)
            ],
        ),
    ):
        size = 8.0 + 22.0 * np.sqrt(mass / max(mass.max(), 1e-12))
        fig.add_trace(
            trace(
                **_xy(xy, inst.geographic),
                mode="markers",
                name=label,
                marker={
                    "size": size,
                    "color": p.categorical[slot],
                    "line": {"width": 2, "color": p.surface},
                    "symbol": "square" if slot == DEPOT else "circle",
                },
                hoverinfo="text",
                text=hover,
            )
        )

    fig.update_layout(
        template=template(dark),
        title=f"Optimal shipments · total cost {sol.total_cost:,.1f} {inst.cost_unit}·units",
        legend={"orientation": "h", "y": -0.02, "x": 0},
        height=620,
    )
    if inst.geographic:
        fig.update_geos(fitbounds="locations", projection_type="mercator", resolution=50)
    else:
        fig.update_xaxes(showgrid=False, showticklabels=False, constrain="domain")
        fig.update_yaxes(showgrid=False, showticklabels=False, scaleanchor="x")
    return fig


def shadow_prices(
    inst: TransportInstance, sol: TransportSolution, *, dark: bool = False
) -> go.Figure:
    """Demand duals: marginal cost of serving one more unit at each customer."""
    if sol.demand_duals is None:
        raise ValueError("solution has no duals (MIP or non-optimal LP)")
    p = palette(dark)
    order = np.argsort(sol.demand_duals)
    names = [inst.sinks[k] for k in order]
    values = sol.demand_duals[order]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker={"color": p.categorical[DEPOT], "line": {"width": 0}, "cornerradius": 4},
            hovertemplate="<b>%{x:,.2f}</b> "
            + inst.cost_unit
            + " per extra unit<br>%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        template=template(dark),
        title="Shadow price of demand (dual v<sub>j</sub>)",
        xaxis_title=f"Marginal cost of one more unit ({inst.cost_unit})",
        yaxis={"showgrid": False},
        height=max(320, 18 * len(names) + 120),
        bargap=0.25,
    )
    return fig


def depot_utilisation(
    inst: TransportInstance, sol: TransportSolution, *, dark: bool = False
) -> go.Figure:
    """Capacity used per depot; hover shows the value of extra capacity (-u_i)."""
    p = palette(dark)
    util = sol.utilisation(inst.supply)
    value = -sol.supply_duals if sol.supply_duals is not None else np.full(len(util), np.nan)
    order = np.argsort(util)
    fig = go.Figure(
        go.Bar(
            x=util[order] * 100,
            y=[inst.sources[k] for k in order],
            orientation="h",
            marker={"color": p.categorical[DEPOT], "line": {"width": 0}, "cornerradius": 4},
            customdata=np.column_stack([inst.supply[order], value[order]]),
            hovertemplate=(
                "<b>%{x:.1f}%</b> of %{customdata[0]:,.1f} used<br>%{y}"
                "<br>extra capacity saves %{customdata[1]:,.2f} "
                + inst.cost_unit
                + "/unit<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template=template(dark),
        title="Depot utilisation",
        xaxis={"title": "Capacity used (%)", "range": [0, 105]},
        yaxis={"showgrid": False},
        height=max(280, 26 * len(order) + 120),
        bargap=0.3,
    )
    return fig


def flow_matrix(
    inst: TransportInstance, sol: TransportSolution, *, dark: bool = False
) -> go.Figure:
    """Heatmap of the transport plan; empty cells are unused routes."""
    z = np.where(sol.flow > 1e-9, sol.flow, np.nan)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(inst.sinks),
            y=list(inst.sources),
            colorscale=sequential_scale(dark),
            hoverongaps=False,
            colorbar={"title": {"text": "flow"}, "thickness": 12, "outlinewidth": 0},
            hovertemplate="<b>%{z:,.1f}</b><br>%{y} → %{x}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        template=template(dark),
        title="Transport plan x<sub>ij</sub>",
        xaxis={"showgrid": False, "tickangle": -45},
        yaxis={"showgrid": False, "autorange": "reversed"},
        height=max(320, 22 * len(inst.sources) + 200),
    )
    return fig


def all_figures(
    inst: TransportInstance, sol: TransportSolution, *, dark: bool = False
) -> list[go.Figure]:
    """Every transport figure, in reading order."""
    figs = [flow_map(inst, sol, dark=dark), depot_utilisation(inst, sol, dark=dark)]
    if sol.demand_duals is not None:
        figs.append(shadow_prices(inst, sol, dark=dark))
    figs.append(flow_matrix(inst, sol, dark=dark))
    return figs
