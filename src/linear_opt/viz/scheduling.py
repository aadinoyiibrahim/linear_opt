"""Figures for job-shop scheduling: Gantt charts with the critical path."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from linear_opt.problems.jobshop import JobShopInstance, JobShopSolution
from linear_opt.viz.mip import mip_progress
from linear_opt.viz.theme import Palette, palette, template

MAX_JOB_COLOURS = 7  # categorical slots validated for adjacent marks


def _job_colour(p: Palette, j: int, n_jobs: int) -> str:
    """Fixed-order categorical hue per job; beyond 7 jobs one hue plus labels."""
    return p.categorical[j] if n_jobs <= MAX_JOB_COLOURS else p.categorical[0]


def _gantt(inst: JobShopInstance, sol: JobShopSolution, *, by: str, dark: bool) -> go.Figure:
    p = palette(dark)
    s, dur, mu = sol.starts.ravel(), inst.p, inst.mu
    critical = set(sol.critical)
    rows = (
        [f"M{m + 1}" for m in range(inst.n_machines)]
        if by == "machine"
        else [f"J{j + 1}" for j in range(inst.n_jobs)]
    )
    fig = go.Figure()
    for j in range(inst.n_jobs):
        ops = np.arange(j * inst.n_ops, (j + 1) * inst.n_ops)
        on_path = np.array([o in critical for o in ops])
        y = [rows[mu[o]] if by == "machine" else rows[j] for o in ops]
        label = [f"J{j + 1}" if by == "machine" else f"M{mu[o] + 1}" for o in ops]
        fig.add_trace(
            go.Bar(
                base=s[ops],
                x=dur[ops],
                y=y,
                orientation="h",
                name=f"job {j + 1}",
                legendgroup=f"job {j + 1}",
                showlegend=False,  # a clean swatch is added below
                marker={
                    "color": _job_colour(p, j, inst.n_jobs),
                    "line": {
                        "color": np.where(on_path, p.ink, p.surface).tolist(),
                        "width": np.where(on_path, 3.0, 1.0).tolist(),
                    },
                    "cornerradius": 3,
                },
                text=label,
                textposition="inside",
                insidetextanchor="middle",
                textangle=0,
                textfont={"color": "#ffffff", "size": 11},
                customdata=np.column_stack(
                    [s[ops] + dur[ops], mu[ops] + 1, np.arange(1, inst.n_ops + 1), on_path]
                ),
                hovertemplate=(
                    f"<b>job {j + 1}</b>, step %{{customdata[2]}} on M%{{customdata[1]}}"
                    "<br>%{base:g} → %{customdata[0]:g} (%{x:g})"
                    "<br>critical: %{customdata[3]}<extra></extra>"
                ),
            )
        )
    # legend swatches without outlines (a bar's first outline would leak into the legend)
    for j in range(inst.n_jobs):
        fig.add_trace(
            go.Bar(
                x=[None],
                y=[rows[0]],
                orientation="h",
                name=f"job {j + 1}",
                legendgroup=f"job {j + 1}",
                marker={"color": _job_colour(p, j, inst.n_jobs), "line": {"width": 0}},
                hoverinfo="skip",
            )
        )
    # legend key for the critical-path outline
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[rows[0]],
            orientation="h",
            name="critical path (outlined)",
            marker={"color": "rgba(0,0,0,0)", "line": {"color": p.ink, "width": 3}},
            hoverinfo="skip",
        )
    )
    fig.add_vline(
        x=sol.makespan,
        line={"dash": "dot", "width": 1.5, "color": p.muted},
        annotation_text=f"makespan {sol.makespan:g}",
        annotation_font={"color": p.ink_secondary, "size": 12},
        annotation_position="top",
    )
    title = (
        f"Schedule by {by} · makespan {sol.makespan:g}"
        + (f" (published optimum {inst.reference:g})" if inst.reference is not None else "")
        + f" · critical path of {len(sol.critical)} operations"
    )
    fig.update_layout(
        template=template(dark),
        title=title,
        barmode="overlay",
        uniformtext={"minsize": 9, "mode": "hide"},  # hide labels that do not fit
        bargap=0.25,
        xaxis={"title": "time", "range": [0, sol.makespan * 1.06], "rangemode": "tozero"},
        yaxis={
            "showgrid": False,
            "autorange": "reversed",
            "categoryorder": "array",
            "categoryarray": rows,
        },
        legend={"orientation": "h", "y": -0.18, "x": 0},
        height=max(320, 46 * len(rows) + 170),
    )
    return fig


def gantt_by_machine(
    inst: JobShopInstance, sol: JobShopSolution, *, dark: bool = False
) -> go.Figure:
    """One row per machine; bars coloured and labelled by job."""
    return _gantt(inst, sol, by="machine", dark=dark)


def gantt_by_job(inst: JobShopInstance, sol: JobShopSolution, *, dark: bool = False) -> go.Figure:
    """One row per job; bars labelled by machine (gaps = waiting time)."""
    return _gantt(inst, sol, by="job", dark=dark)


def machine_utilisation(
    inst: JobShopInstance, sol: JobShopSolution, *, dark: bool = False
) -> go.Figure:
    """Busy time of each machine as a share of the makespan."""
    p = palette(dark)
    busy = np.bincount(inst.mu, weights=inst.p, minlength=inst.n_machines)
    pct = 100 * busy / max(sol.makespan, 1e-12)
    names = [f"M{m + 1}" for m in range(inst.n_machines)]
    fig = go.Figure(
        go.Bar(
            x=pct,
            y=names,
            orientation="h",
            marker={"color": p.categorical[0], "line": {"width": 0}, "cornerradius": 4},
            customdata=busy,
            hovertemplate="<b>%{x:.1f}%</b> busy (%{customdata:g} time units)<br>%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        template=template(dark),
        title="Machine utilisation (the busiest machine bounds the makespan from below)",
        xaxis={"title": "busy share of makespan (%)", "range": [0, 105]},
        yaxis={"showgrid": False, "autorange": "reversed"},
        height=max(240, 30 * len(names) + 120),
        bargap=0.3,
    )
    return fig


def all_figures(
    inst: JobShopInstance, sol: JobShopSolution, *, dark: bool = False
) -> list[go.Figure]:
    """Every scheduling figure, in reading order."""
    figs = [
        gantt_by_machine(inst, sol, dark=dark),
        gantt_by_job(inst, sol, dark=dark),
        machine_utilisation(inst, sol, dark=dark),
    ]
    if sol.progress:
        figs.append(mip_progress(sol.progress, reference=inst.reference, dark=dark))
    return figs
