"""linear_opt dashboard.

Run with ``uv run linopt app`` (or ``streamlit run src/linear_opt/app/main.py``).
All logic lives in :mod:`linear_opt.app.logic`; this file only draws widgets.

"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from linear_opt.app import logic
from linear_opt.core.backends import available_backends
from linear_opt.core.config import ProblemKind, RunConfig

st.set_page_config(page_title="linear_opt", page_icon=":material/hub:", layout="wide")

COLUMN_NAMES = {
    "lp_bound": "LP bound",
    "root_gap": "root gap (%)",
    "runtime_s": "runtime (s)",
    "constraints": "rows",
    "variables": "columns",
}


# ------------------------------------------------------------------ widgets
def render_specs(
    specs: list[logic.WidgetSpec], defaults: dict[str, Any], prefix: str
) -> dict[str, Any]:
    """Draw one input per spec; ``prefix`` scopes widget state to the current preset."""
    values: dict[str, Any] = {}
    for spec in specs:
        label = spec.name.replace("_", " ")
        default = defaults.get(spec.name, spec.default)
        key = f"{prefix}:{spec.name}"
        if spec.kind == "select":
            choices = list(spec.choices)
            index = choices.index(default) if default in choices else 0
            values[spec.name] = st.selectbox(
                label, choices, index=index, key=key, help=spec.help or None
            )
        elif spec.kind == "bool":
            values[spec.name] = st.checkbox(
                label, value=bool(default), key=key, help=spec.help or None
            )
        elif spec.kind == "int":
            lo = None if spec.minimum is None else int(spec.minimum)
            hi = None if spec.maximum is None else int(spec.maximum)
            values[spec.name] = int(
                st.number_input(
                    label, value=int(default), min_value=lo, max_value=hi, step=1, key=key
                )
            )
        elif spec.kind == "float":
            values[spec.name] = float(
                st.number_input(
                    label,
                    value=float(default),
                    min_value=spec.minimum,
                    max_value=spec.maximum,
                    key=key,
                )
            )
        else:
            values[spec.name] = st.text_input(label, value=str(default), key=key)
    return values


def sidebar() -> tuple[ProblemKind, RunConfig | None, bool]:
    """All settings; returns the problem, a validated config (or None) and the chart theme."""
    sb = st.sidebar
    sb.title("linear_opt")
    sb.caption("LP / MILP with Gurobi and HiGHS on benchmark data")
    kind: ProblemKind = sb.selectbox(
        "Problem", list(logic.PROBLEMS), format_func=lambda k: logic.PROBLEMS[k], key="kind"
    )
    presets = logic.presets(kind)
    names = [*presets, "(custom)"]
    preset_name = sb.selectbox("Preset", names, key=f"preset:{kind.value}")
    base = presets.get(preset_name)
    prefix = f"{kind.value}:{preset_name}"

    choices = logic.instance_choices(kind)
    if not choices:
        sb.error("No instance available. Download data with `uv run linopt data ...`.")
        return kind, None, False
    sources = list(choices)
    default_source = (
        base.instance.source if base and base.instance.source in choices else sources[0]
    )

    with sb.expander("Instance", expanded=True):
        source = st.selectbox(
            "Source",
            sources,
            index=sources.index(default_source),
            format_func=lambda s: logic.SOURCE_LABELS.get(s, s),
            key=f"{prefix}:source",
        )
        ids = choices[source]
        default_id = base.instance.id if base and base.instance.id in ids else ids[0]
        instance_id = st.selectbox(
            "Instance", ids, index=ids.index(default_id), key=f"{prefix}:{source}:id"
        )
        seed = int(base.instance.seed) if base else 0
        if source == "synthetic":
            seed = int(
                st.number_input(
                    "Random seed", value=seed, min_value=0, step=1, key=f"{prefix}:seed"
                )
            )
        same_source = base is not None and base.instance.source == source
        params = render_specs(
            logic.widget_specs(logic.params_model(kind, source)),
            dict(base.instance.params) if same_source and base else {},
            f"{prefix}:{source}:params",
        )

    with sb.expander("Model", expanded=True):
        options = render_specs(
            logic.widget_specs(logic.options_model(kind)),
            dict(base.model) if base else {},
            f"{prefix}:model",
        )

    with sb.expander("Solver", expanded=False):
        backends = ["auto", *(b.value for b in available_backends())]
        default_backend = base.solver.backend.value if base else "auto"
        backend = st.selectbox(
            "Backend", backends, index=backends.index(default_backend) if default_backend in backends else 0,
            key=f"{prefix}:backend",
        )  # fmt: skip
        time_limit = st.number_input(
            "Time limit (s)", value=float(base.solver.time_limit if base else 60.0), min_value=1.0,
            max_value=3600.0, key=f"{prefix}:time_limit",
        )  # fmt: skip
        gap_choices = [0.0, 1e-4, 1e-3, 1e-2, 5e-2]
        default_gap = base.solver.mip_gap if base else 1e-4
        mip_gap = st.select_slider(
            "Relative MIP gap", gap_choices, value=min(gap_choices, key=lambda g: abs(g - default_gap)),
            format_func=lambda g: f"{100 * g:g}%", key=f"{prefix}:gap",
        )  # fmt: skip
        native = (
            {"gurobi": dict(base.solver.gurobi), "highs": dict(base.solver.highs)} if base else {}
        )
        if any(native.values()):
            st.caption(f"Native parameters from the preset: {json.dumps(native)}")

    dark = sb.toggle("Dark charts", value=False, key="dark")
    try:
        cfg = logic.build_config(
            kind,
            base.problem.name if base else f"custom-{kind.value}",
            source=source,
            instance_id=instance_id,
            seed=seed,
            params=params,
            options=options,
            solver={"backend": backend, "time_limit": time_limit, "mip_gap": mip_gap, **native},
        )
    except ValueError as exc:
        sb.error(f"Invalid settings: {exc}")
        return kind, None, dark
    return kind, cfg, dark


# -------------------------------------------------------------------- views
@st.cache_resource(show_spinner="Loading instance ...", max_entries=16)
def preview(cfg_json: str) -> logic.Outcome:
    """Build (but do not solve) the model; cached per instance and options."""
    return logic.build_model(RunConfig.model_validate_json(cfg_json))


def chart(fig: Any, dark: bool) -> None:
    """Render a figure with our own template (not Streamlit's theme).

    Backgrounds are pinned explicitly: otherwise Streamlit paints the chart with
    its secondary background colour.
    """
    from linear_opt.viz.theme import palette

    surface = palette(dark).surface
    fig.update_layout(paper_bgcolor=surface, plot_bgcolor=surface)
    st.plotly_chart(fig, width="stretch", theme=None)


def stored_result(name: str, key: str) -> Any:
    """A stored result for the current problem kind (``None`` if absent or for another kind).

    Results stay in the session while settings change, so that the page can
    say they are stale - but a result of a different problem type is never shown.
    """
    stored = st.session_state.get(f"result:{name}")
    if stored is None:
        return None
    stored_key, value = stored
    same_kind = json.loads(stored_key)["problem"]["kind"] == json.loads(key)["problem"]["kind"]
    if not same_kind:
        return None
    if stored_key != key:
        st.warning("Settings changed since this result was computed.", icon=":material/history:")
    return value


def _fmt(value: Any) -> str:
    """Readable numbers: thousands separators, no scientific notation for objectives."""
    if value is None:
        return "—"
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-6 * max(1.0, abs(value)):
            return f"{round(value):,}"
        return f"{value:,.3f}"
    return str(value)


def show_solution(kind: ProblemKind, cfg: RunConfig, key: str, dark: bool) -> None:
    """Solve button, KPIs, validation status, figures and downloads."""
    if st.button("Solve", type="primary", key="solve", icon=":material/play_arrow:"):
        with st.spinner("Solving ..."):
            st.session_state["result:solve"] = (key, logic.solve(cfg))
    outcome = stored_result("solve", key)
    if outcome is None:
        st.info("Adjust the settings in the sidebar, then press **Solve**.")
        return
    if outcome.error:
        st.error(outcome.error)
        return
    result = outcome.result
    s = result.summary()
    cols = st.columns(6)
    cols[0].metric("Status", s["status"])
    cols[1].metric("Objective", _fmt(s["objective"]))
    cols[2].metric("Gap", "—" if s["gap"] is None else f"{100 * s['gap']:.3g}%")
    cols[3].metric("Runtime", f"{s['runtime_s']:.3g} s")
    cols[4].metric("Nodes", f"{s['nodes']:,}")
    cols[5].metric("Backend", result.raw.backend, help=f"version {result.raw.backend_version}")
    if s.get("reference") is not None:
        dev = s["reference_gap"]
        st.caption(f"Published optimum **{s['reference']:,g}** · deviation {dev:.2e}")
    if result.violations:
        st.error("Independent validation failed:\n\n- " + "\n- ".join(result.violations[:10]))
    elif result.solution is not None:
        st.success(
            "Solution re-checked against the raw data: all constraints hold.",
            icon=":material/verified:",
        )
    if result.solution is None:
        return

    from linear_opt.viz import figures_for

    figures = figures_for(outcome.model.kind.value, outcome.model.data, result.solution, dark=dark)
    for fig in figures:
        chart(fig, dark)

    from linear_opt.viz.report import model_subtitle, run_kpis, write_report

    with tempfile.TemporaryDirectory() as tmp:
        path = write_report(
            Path(tmp) / "report.html", f"{cfg.problem.name} · {kind.value}", figures,
            kpis=run_kpis(s), subtitle=model_subtitle(s),
        )  # fmt: skip
        html = path.read_bytes()
    c1, c2 = st.columns(2)
    c1.download_button(
        "HTML report", html, f"{cfg.problem.name}.html", "text/html", icon=":material/download:"
    )
    c2.download_button(
        "Summary (JSON)", json.dumps(s, indent=2), f"{cfg.problem.name}.json", "application/json",
        icon=":material/data_object:",
    )  # fmt: skip


def show_compare(cfg: RunConfig, key: str) -> None:
    """Same model on every installed backend."""
    st.write("Solve the current model with every installed backend and compare.")
    if st.button("Run all backends", key="compare", icon=":material/compare_arrows:"):
        with st.spinner("Solving with each backend ..."):
            st.session_state["result:compare"] = (key, logic.compare_backends(cfg))
    rows = stored_result("compare", key)
    if rows is None:
        return
    st.dataframe(pd.DataFrame(rows).rename(columns=COLUMN_NAMES), hide_index=True, width="stretch")
    objectives = [r["objective"] for r in rows if r.get("objective") is not None]
    if len(objectives) > 1:
        spread = (max(objectives) - min(objectives)) / max(1.0, abs(objectives[0]))
        if spread < 1e-6:
            st.success("All backends agree on the objective.")
        else:
            st.warning(f"Objectives differ by {spread:.2e} (relative).")


def show_variants(cfg: RunConfig, key: str, dark: bool) -> None:
    """Formulation study: LP bound, root gap, effort."""
    from linear_opt.core.study import run_variants
    from linear_opt.problems import registry
    from linear_opt.solvers import LicenseLimitError

    variants = registry.MODEL_CLASSES[cfg.problem.kind].variants()
    if not variants:
        st.info("This problem has a single formulation.")
        return
    st.write(
        "Each formulation is solved twice: its **LP relaxation** (with lazy constraints "
        "applied by row generation) and the **MIP**. The root gap measures how tight it is."
    )
    if st.button("Compare formulations", key="variants", icon=":material/science:"):
        with st.spinner(f"Solving {len(variants)} formulations ..."):
            built = logic.build_model(cfg)
            if built.model is None:
                st.session_state["result:variants"] = (key, built.error)
            else:
                try:
                    st.session_state["result:variants"] = (
                        key,
                        run_variants(built.model, cfg.solver),
                    )
                except LicenseLimitError as exc:
                    st.session_state["result:variants"] = (
                        key,
                        f"{exc} Compact formulations grow as n².",
                    )
    results = stored_result("variants", key)
    if results is None:
        return
    if isinstance(results, str):
        st.error(results)
        return
    table = pd.DataFrame([r.row() for r in results])
    table["root_gap"] = (100 * table["root_gap"]).round(3)
    table["lp_bound"] = table["lp_bound"].round(3)
    table["runtime_s"] = table["runtime_s"].round(3)
    st.dataframe(table.rename(columns=COLUMN_NAMES), hide_index=True, width="stretch")
    from linear_opt.viz import mip

    for fig in [mip.variant_root_gaps(results, dark=dark), *mip.variant_effort(results, dark=dark)]:
        chart(fig, dark)
    chart(mip.variant_progress(results, dark=dark), dark)


def show_model(cfg: RunConfig, built: logic.Outcome) -> None:
    """Model structure and exports (LP file, reproducible TOML config)."""
    assert built.model is not None
    lm = built.model.build()
    left, right = st.columns(2)
    with left:
        st.subheader("Variable blocks")
        st.dataframe(
            pd.DataFrame([{"block": b.name, "shape": "×".join(map(str, b.shape)), "size": b.size} for b in lm.var_blocks.values()]),
            hide_index=True, width="stretch",
        )  # fmt: skip
    with right:
        st.subheader("Constraint blocks")
        st.dataframe(
            pd.DataFrame([{"block": b.name, "rows": b.size, "sense": b.sense.value} for b in lm.constr_blocks.values()]),
            hide_index=True, width="stretch",
        )  # fmt: skip
    st.caption(
        "Lazy constraints and cuts are added during the solve and are not part of these blocks."
    )

    st.subheader("Reproduce from the command line")
    toml = logic.to_toml(cfg)
    st.code(toml, language="toml")
    c1, c2 = st.columns(2)
    c1.download_button(
        "Config (TOML)",
        toml,
        f"{cfg.problem.name}.toml",
        "application/toml",
        icon=":material/settings:",
    )
    try:
        lp = logic.export_model(built.model, cfg.solver.backend.value)
        c2.download_button(
            "Model (.lp)", lp, f"{cfg.problem.name}.lp", "text/plain", icon=":material/description:"
        )
    except Exception as exc:
        c2.caption(f"LP export unavailable: {exc}")
    st.caption(f"Then run: `uv run linopt solve {cfg.problem.name}.toml --html report.html`")


# --------------------------------------------------------------------- page
def main() -> None:
    """Draw the page."""
    kind, cfg, dark = sidebar()
    st.title(logic.PROBLEMS[kind])
    if cfg is None:
        return
    key = cfg.model_dump_json()
    built = preview(cfg.model_copy(update={"solver": type(cfg.solver)()}).model_dump_json())
    if built.model is None:
        st.error(built.error)
        return
    stats = built.model.build().stats()
    source = logic.SOURCE_LABELS.get(cfg.instance.source or "", cfg.instance.source)
    st.caption(f"{source} · **{cfg.instance.id}** · {cfg.problem.name}")
    cols = st.columns(4)
    cols[0].metric("Variables", f"{stats['variables']:,}")
    cols[1].metric("Integer variables", f"{stats['integer_variables']:,}")
    cols[2].metric("Constraints", f"{stats['constraints']:,}")
    cols[3].metric("Non-zeros", f"{stats['nonzeros']:,}")
    if warning := logic.licence_warning(built.model, cfg.solver.backend.value):
        st.warning(warning, icon=":material/warning:")

    solution, compare, formulations, model = st.tabs(
        ["Solution", "Compare backends", "Formulations", "Model & export"]
    )
    with solution:
        show_solution(kind, cfg, key, dark)
    with compare:
        show_compare(cfg, key)
    with formulations:
        show_variants(cfg, key, dark)
    with model:
        show_model(cfg, built)


main()
