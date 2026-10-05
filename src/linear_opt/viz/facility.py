"""Figures for capacitated facility location."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from linear_opt.problems.facility import FacilityInstance, FacilitySolution
from linear_opt.viz.mip import mip_progress
from linear_opt.viz.theme import palette, sequential_scale, template

FACILITY, CUSTOMER = 0, 1  # categorical slots: identity colours shared across figures


def facility_map(inst: FacilityInstance, sol: FacilitySolution, *, dark: bool = False) -> go.Figure:
    """Open/closed sites, customers and assignments on a map (geographic instances only)."""
    if inst.facility_xy is None or inst.customer_xy is None:
        raise ValueError("instance has no coordinates")
    p = palette(dark)
    geo = inst.geographic
    trace = go.Scattergeo if geo else go.Scatter

    def xy(points: np.ndarray) -> dict[str, np.ndarray]:
        return (
            {"lat": points[:, 0], "lon": points[:, 1]}
            if geo
            else {"x": points[:, 0], "y": points[:, 1]}
        )

    fig = go.Figure()
    flows = sol.assignment * inst.demand[None, :]
    peak = float(flows.max(initial=1.0))
    for i, j in zip(*np.nonzero(sol.assignment > 1e-6), strict=True):
        share = sol.assignment[i, j]
        fig.add_trace(
            trace(
                **xy(np.vstack([inst.facility_xy[i], inst.customer_xy[j]])),
                mode="lines",
                line={"width": 0.8 + 4.0 * flows[i, j] / peak, "color": p.ink_secondary},
                opacity=0.5,
                hoverinfo="text",
                text=f"{inst.customers[j]} ← {inst.facilities[i]}"
                f"<br><b>{100 * share:.0f}%</b> of demand",
                showlegend=False,
            )
        )
    load = sol.load(inst.demand)
    fig.add_trace(
        trace(
            **xy(inst.customer_xy),
            mode="markers",
            name="Customer (size = demand)",
            marker={
                "size": 6 + 18 * np.sqrt(inst.demand / inst.demand.max()),
                "color": p.categorical[CUSTOMER],
                "line": {"width": 2, "color": p.surface},
            },
            hoverinfo="text",
            text=[
                f"<b>{n}</b><br>demand {d:,.1f}"
                for n, d in zip(inst.customers, inst.demand, strict=True)
            ],
        )
    )
    for is_open, label in ((False, "Candidate site (closed)"), (True, "Open facility")):
        idx = np.flatnonzero(sol.open == is_open)
        if idx.size == 0:
            continue
        fig.add_trace(
            trace(
                **xy(inst.facility_xy[idx]),
                mode="markers",
                name=label,
                marker={
                    "size": 16 if is_open else 11,
                    "symbol": "square" if is_open else "square-open",
                    "color": p.categorical[FACILITY] if is_open else p.muted,
                    "line": {"width": 2, "color": p.surface if is_open else p.muted},
                },
                hoverinfo="text",
                text=[
                    f"<b>{inst.facilities[i]}</b><br>"
                    + (f"load {load[i]:,.0f} / {inst.capacity[i]:,.0f}" if is_open else "closed")
                    + f"<br>opening cost {inst.fixed_cost[i]:,.0f}"
                    for i in idx
                ],
            )
        )
    fig.update_layout(
        template=template(dark),
        title=f"{int(sol.open.sum())} of {len(inst.facilities)} sites open"
        f" · total cost {sol.total_cost:,.0f}",
        legend={"orientation": "h", "y": -0.02, "x": 0},
        height=640,
    )
    if geo:
        fig.update_geos(fitbounds="locations", projection_type="mercator", resolution=50)
    else:
        fig.update_xaxes(showgrid=False, showticklabels=False, constrain="domain")
        fig.update_yaxes(showgrid=False, showticklabels=False, scaleanchor="x")
    return fig


def _open_order(inst: FacilityInstance, sol: FacilitySolution) -> np.ndarray:
    idx = np.flatnonzero(sol.open)
    return idx[np.argsort(sol.load(inst.demand)[idx] / inst.capacity[idx])]


def utilisation(inst: FacilityInstance, sol: FacilitySolution, *, dark: bool = False) -> go.Figure:
    """Capacity used at each open facility."""
    p = palette(dark)
    idx = _open_order(inst, sol)
    load = sol.load(inst.demand)
    pct = 100 * load[idx] / inst.capacity[idx]
    fig = go.Figure(
        go.Bar(
            x=pct,
            y=[inst.facilities[i] for i in idx],
            orientation="h",
            marker={"color": p.categorical[FACILITY], "line": {"width": 0}, "cornerradius": 4},
            customdata=np.column_stack([load[idx], inst.capacity[idx]]),
            hovertemplate="<b>%{x:.1f}%</b> used<br>%{customdata[0]:,.0f} of %{customdata[1]:,.0f}"
            "<br>%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        template=template(dark),
        title="Utilisation of open facilities",
        xaxis={"title": "capacity used (%)", "range": [0, 105]},
        yaxis={"showgrid": False},
        height=max(260, 28 * len(idx) + 120),
        bargap=0.3,
    )
    return fig


def cost_breakdown(
    inst: FacilityInstance, sol: FacilitySolution, *, dark: bool = False
) -> go.Figure:
    """Opening cost vs. serving cost per open facility (stacked)."""
    p = palette(dark)
    idx = _open_order(inst, sol)
    names = [inst.facilities[i] for i in idx]
    serving = (inst.assign_cost * sol.assignment).sum(axis=1)
    fig = go.Figure()
    for label, values, slot in (
        ("opening cost", inst.fixed_cost[idx], 0),
        ("serving cost", serving[idx], 1),
    ):
        fig.add_trace(
            go.Bar(
                x=values,
                y=names,
                name=label,
                orientation="h",
                marker={"color": p.categorical[slot], "line": {"width": 2, "color": p.surface}},
                hovertemplate="<b>%{x:,.0f}</b> " + label + "<br>%{y}<extra></extra>",
            )
        )
    share = 100 * sol.fixed_part / max(sol.total_cost, 1e-12)
    fig.update_layout(
        template=template(dark),
        barmode="stack",
        title=f"Cost per open facility · opening {share:.0f}% / serving {100 - share:.0f}%",
        xaxis={"title": "cost", "rangemode": "tozero"},
        yaxis={"showgrid": False},
        legend={
            "orientation": "h",
            "y": 1.02,
            "yanchor": "bottom",
            "x": 1,
            "xanchor": "right",
            "traceorder": "normal",
        },
        height=max(280, 28 * len(idx) + 140),
        bargap=0.3,
    )
    return fig


def assignment_matrix(
    inst: FacilityInstance, sol: FacilitySolution, *, dark: bool = False
) -> go.Figure:
    """Share of each customer's demand served by each open facility."""
    idx = np.flatnonzero(sol.open)
    z = np.where(sol.assignment[idx] > 1e-6, 100 * sol.assignment[idx], np.nan)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(inst.customers),
            y=[inst.facilities[i] for i in idx],
            zmin=0,
            zmax=100,
            colorscale=sequential_scale(dark),
            hoverongaps=False,
            xgap=2,
            ygap=2,
            colorbar={"title": {"text": "% served"}, "thickness": 12, "outlinewidth": 0},
            hovertemplate="<b>%{z:.0f}%</b> of %{x}<br>served by %{y}<extra></extra>",
        )
    )
    split = int(((sol.assignment > 1e-6).sum(axis=0) > 1).sum())
    fig.update_layout(
        template=template(dark),
        title=f"Assignments · {split} of {len(inst.customers)} customers split across sites",
        xaxis={"showgrid": False, "tickangle": -45},
        yaxis={"showgrid": False},
        height=max(300, 24 * len(idx) + 200),
    )
    return fig


def all_figures(
    inst: FacilityInstance, sol: FacilitySolution, *, dark: bool = False
) -> list[go.Figure]:
    """Every facility figure, in reading order."""
    figs = []
    if inst.facility_xy is not None and inst.customer_xy is not None:
        figs.append(facility_map(inst, sol, dark=dark))
    figs += [cost_breakdown(inst, sol, dark=dark), utilisation(inst, sol, dark=dark)]
    if sol.progress:
        figs.append(mip_progress(sol.progress, reference=inst.reference, dark=dark))
    figs.append(assignment_matrix(inst, sol, dark=dark))
    return figs
