"""Render the README animation: the optimal transport plan between German cities.

Solves ``configs/transport_de_cities.toml`` and draws the shipments one route at
a time, largest first, with the running cost; the last frame states the
optimum and the independent feasibility check. Figures come from
``linearmilpopt.viz.transport.flow_map``, so the GIF matches the dashboard.

    uv run --with kaleido python scripts/make_readme_gif.py

Static export needs kaleido (not a project dependency) and Chrome; if Chrome is
missing, run ``uv run --with kaleido plotly_get_chrome`` once.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image

from linearmilpopt.core.config import load_config
from linearmilpopt.problems.registry import model_from_config
from linearmilpopt.problems.transport import TransportInstance, TransportSolution
from linearmilpopt.viz.theme import palette
from linearmilpopt.viz.transport import flow_map

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "transport_de_cities.toml"
OUT = ROOT / "docs" / "assets" / "transport_flows.gif"
SIZE = 720  # px, square: Germany is taller than wide


def frames(
    fig: go.Figure,
    inst: TransportInstance,
    sol: TransportSolution,
    backend: str,
    runtime: float,
) -> list[go.Figure]:
    """One figure per frame: cities only, then routes added largest first."""
    p = palette(False)
    routes = sol.routes()
    n_routes = len(routes)
    order = sorted(range(n_routes), key=lambda k: -routes[k][2])  # route traces come first
    unit = f"{inst.cost_unit}·units"

    def styled(title: str, shown: list[int], newest: int | None) -> go.Figure:
        f = go.Figure(fig)
        for k in range(n_routes):
            tr = f.data[k]
            tr.visible = k in shown
            if k == newest:
                tr.line.color, tr.opacity = p.categorical[3], 0.95  # amber: just added
            else:
                tr.line.color, tr.opacity = p.ink_secondary, 0.55
        f.update_layout(title={"text": title, "x": 0.02, "font": {"size": 17}})
        return f

    out = [
        styled(
            f"{len(inst.sinks)} cities · {len(inst.sources)} depots · "
            f"LP with {len(inst.sources) * len(inst.sinks)} variables",
            [],
            None,
        )
    ]
    cost = 0.0
    for step, k in enumerate(order, 1):
        i, j, flow = routes[k]
        cost += flow * inst.cost[i, j]
        out.append(
            styled(
                f"Route {step}/{n_routes}: {inst.sources[i]} → {inst.sinks[j]}"
                f"<br><sup>cost so far {cost:,.0f} {unit}</sup>",
                order[:step],
                k,
            )
        )
    if not np.isclose(cost, sol.total_cost, rtol=1e-9):
        raise RuntimeError(f"route costs {cost} do not add up to the optimum {sol.total_cost}")
    out.append(
        styled(
            f"Optimal plan: {sol.total_cost:,.0f} {unit}"
            f"<br><sup>{backend} · {runtime * 1000:.1f} ms · re-checked feasible</sup>",
            order,
            None,
        )
    )
    return out


def main() -> None:
    """Solve, render every frame and write the GIF."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("-o", "--output", type=Path, default=OUT)
    ap.add_argument("--topojson", help="base URL of Plotly's map files (offline use)")
    args = ap.parse_args()
    if args.topojson:
        pio.defaults.topojson = args.topojson

    cfg = load_config(CONFIG)
    model = model_from_config(cfg)
    result = model.solve(cfg.solver)
    if result.violations:
        raise RuntimeError(f"solution failed the independent check: {result.violations}")
    inst, sol = model.data, result.solution
    if sol is None:
        raise RuntimeError(f"no solution: {result.summary()['status']}")
    summary = result.summary()

    base = flow_map(inst, sol)
    base.update_layout(
        width=SIZE,
        height=SIZE,
        margin={"l": 10, "r": 10, "t": 70, "b": 40},
        legend={"orientation": "h", "y": 0.0, "x": 0.02},
    )
    figs = frames(base, inst, sol, summary["backend"], summary["runtime_s"])

    images = [
        Image.open(io.BytesIO(f.to_image(format="png", width=SIZE, height=SIZE))).convert("RGB")
        for f in figs
    ]
    # One shared palette keeps colours stable between frames and the file small.
    # Build it from a frame with the amber highlight and the final frame, so
    # the highlight colour survives quantisation.
    sample = Image.new("RGB", (2 * SIZE, SIZE))
    sample.paste(images[1], (0, 0))
    sample.paste(images[-1], (SIZE, 0))
    pal = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    gif = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in images]
    durations = [1200] + [180] * (len(gif) - 2) + [3500]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gif[0].save(
        args.output,
        save_all=True,
        append_images=gif[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    kb = args.output.stat().st_size / 1024
    print(f"wrote {args.output.relative_to(ROOT)}: {len(gif)} frames, {kb:,.0f} KB")


if __name__ == "__main__":
    main()
