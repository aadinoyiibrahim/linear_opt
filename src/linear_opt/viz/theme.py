"""One visual system for every figure: palette, typography and Plotly templates.

Colours follow a CVD-validated categorical order (slots are assigned in fixed
order, never cycled), a single-hue sequential ramp for magnitudes, and recessive
chrome (hairline grids, muted axes) so the data carries the ink. Both a light
and a dark template are provided; dark steps are chosen for the dark surface,
not inverted.
"""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go
import plotly.io as pio


@dataclass(frozen=True)
class Palette:
    """Colour roles for one mode."""

    surface: str
    page: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    axis: str
    categorical: tuple[str, ...]
    sequential: tuple[str, ...]
    land: str
    border: str


LIGHT = Palette(
    surface="#fcfcfb",
    page="#f9f9f7",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    categorical=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"),
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
    land="#f0efec",
    border="rgba(11,11,11,0.10)",
)
DARK = Palette(
    surface="#1a1a19",
    page="#0d0d0d",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    categorical=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"),
    sequential=("#104281", "#184f95", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#cde2fb"),
    land="#262624",
    border="rgba(255,255,255,0.10)",
)
FONT = "Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


def palette(dark: bool = False) -> Palette:
    """The palette for the requested mode."""
    return DARK if dark else LIGHT


def _template(p: Palette) -> go.layout.Template:
    axis = {
        "gridcolor": p.grid,
        "gridwidth": 1,
        "linecolor": p.axis,
        "zeroline": False,
        "ticks": "",
        "tickfont": {"color": p.muted, "size": 12},
        "title": {"font": {"color": p.ink_secondary, "size": 13}},
        "automargin": True,
        "ticklabelstandoff": 6,
    }
    return go.layout.Template(
        layout={
            "font": {"family": FONT, "color": p.ink, "size": 13},
            "paper_bgcolor": p.surface,
            "plot_bgcolor": p.surface,
            "colorway": list(p.categorical),
            "title": {"font": {"size": 16, "color": p.ink}, "x": 0, "xanchor": "left"},
            "xaxis": axis,
            "yaxis": axis,
            "legend": {"font": {"color": p.ink_secondary}, "bgcolor": "rgba(0,0,0,0)"},
            "hoverlabel": {
                "bgcolor": p.surface,
                "bordercolor": p.axis,
                "font": {"family": FONT, "color": p.ink, "size": 13},
            },
            "margin": {"l": 16, "r": 16, "t": 56, "b": 16},
            "colorscale": {"sequential": [[i / 6, c] for i, c in enumerate(p.sequential)]},
            "geo": {
                "bgcolor": p.surface,
                "landcolor": p.land,
                "showland": True,
                "showcountries": True,
                "countrycolor": p.axis,
                "showcoastlines": True,
                "coastlinecolor": p.axis,
                "showlakes": False,
                "showframe": False,
            },
        }
    )


TEMPLATE_LIGHT = "linopt"
TEMPLATE_DARK = "linopt_dark"
pio.templates[TEMPLATE_LIGHT] = _template(LIGHT)
pio.templates[TEMPLATE_DARK] = _template(DARK)


def template(dark: bool = False) -> str:
    """Registered template name for the mode."""
    return TEMPLATE_DARK if dark else TEMPLATE_LIGHT


def sequential_scale(dark: bool = False) -> list[list[float | str]]:
    """Plotly colourscale built from the sequential ramp (low -> high)."""
    ramp = palette(dark).sequential
    return [[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)]
