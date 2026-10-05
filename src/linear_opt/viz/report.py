"""Write a self-contained HTML report: KPI tiles plus interactive figures."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from linear_opt.viz.theme import FONT, LIGHT


def fmt(value: Any) -> str:
    """Compact human formatting for KPI values."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != 0 and abs(value) < 1e-2:
            return f"{value:.2e}"
        return f"{value:,.2f}" if abs(value) < 1e6 else f"{value:,.0f}"
    return "—" if value is None else str(value)


def run_kpis(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Standard tiles for a single solve."""
    tiles: dict[str, Any] = {
        "Status": summary.get("status"),
        "Objective": summary.get("objective"),
        "Gap": summary.get("gap"),
        "Runtime (s)": summary.get("runtime_s"),
        "Backend": summary.get("backend"),
    }
    if summary.get("reference") is not None:
        tiles["Known optimum"] = summary["reference"]
        tiles["Deviation"] = summary.get("reference_gap")
    return tiles


def model_subtitle(summary: Mapping[str, Any]) -> str:
    """One-line model size description."""
    stats = summary.get("model", {})
    return ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in stats.items())


def write_report(
    path: Path,
    title: str,
    figures: Sequence[go.Figure],
    *,
    kpis: Mapping[str, Any],
    subtitle: str = "",
) -> Path:
    """Render KPI tiles and ``figures`` into one HTML file (Plotly loaded from CDN)."""
    p = LIGHT
    tiles = "".join(
        f'<div class="kpi"><div class="v">{html.escape(fmt(value))}</div>'
        f'<div class="l">{html.escape(label)}</div></div>'
        for label, value in kpis.items()
    )
    sections = [
        fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False, auto_play=False)
        for i, fig in enumerate(figures)
    ]
    body = "".join(f"<section>{s}</section>" for s in sections)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
body{{margin:0;background:{p.page};color:{p.ink};font-family:{FONT}}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 4px}}
.sub{{color:{p.ink_secondary};font-size:13px;margin-bottom:16px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:12px;margin-bottom:16px}}
.kpi{{background:{p.surface};border:1px solid {p.border};border-radius:8px;
  padding:12px 14px}}
.kpi .v{{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}}
.kpi .l{{color:{p.ink_secondary};font-size:12px;margin-top:2px}}
section{{background:{p.surface};border:1px solid {p.border};border-radius:8px;
  margin-bottom:16px;padding:8px}}
</style></head><body><main>
<h1>{html.escape(title)}</h1><div class="sub">{html.escape(subtitle)}</div>
<div class="kpis">{tiles}</div>{body}</main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path
