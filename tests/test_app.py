"""Dashboard: pure logic (unit tests) and the real page (Streamlit AppTest)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from linear_opt.app import logic
from linear_opt.core.config import ProblemKind, RunConfig, load_config
from linear_opt.problems import registry

from .conftest import CONFIG_DIR, INSTALLED

pytest.importorskip("pandas")


# ------------------------------------------------------------------- logic
def test_every_problem_has_presets_and_instances() -> None:
    for kind in ProblemKind:
        assert logic.presets(kind, CONFIG_DIR), kind
        choices = logic.instance_choices(kind)
        assert "synthetic" in choices and choices["synthetic"] == ["uniform"]


def test_presets_skip_invalid_files(tmp_path: Path) -> None:
    (tmp_path / "broken.toml").write_text("[problem\n")
    (tmp_path / "ok.toml").write_text((CONFIG_DIR / "tsp_synthetic.toml").read_text())
    assert list(logic.presets(ProblemKind.TSP, tmp_path)) == ["ok"]
    assert logic.presets(ProblemKind.CVRP, tmp_path) == {}


def test_widget_specs_from_options_and_params() -> None:
    specs = {s.name: s for s in logic.widget_specs(logic.options_model(ProblemKind.JOB_SHOP))}
    assert specs["formulation"].kind == "select"
    assert specs["formulation"].choices == ("disjunctive", "time_indexed")
    params = {
        s.name: s for s in logic.widget_specs(logic.params_model(ProblemKind.CVRP, "synthetic"))
    }
    assert params["n_customers"].kind == "int" and params["n_customers"].minimum == 1
    capacity = params["capacity"]
    assert capacity.kind == "float" and capacity.minimum is not None and capacity.minimum > 0
    bools = {s.name: s for s in logic.widget_specs(logic.options_model(ProblemKind.TSP))}
    assert bools["root_cuts"].kind == "bool" and bools["root_cuts"].default is True
    assert logic.widget_specs(None) == []


def test_build_config_validates_every_section() -> None:
    cfg = logic.build_config(
        ProblemKind.TSP, "t", source="synthetic", instance_id="uniform", seed=3,
        params={"n": 12}, options={"formulation": "mtz"}, solver={"time_limit": 5},
    )  # fmt: skip
    assert cfg.instance.params == {"n": 12} and cfg.model["formulation"] == "mtz"
    with pytest.raises(ValueError, match="n"):
        logic.build_config(
            ProblemKind.TSP, "t", source="synthetic", instance_id="u", params={"n": 1}
        )
    with pytest.raises(ValueError, match="formulation"):
        logic.build_config(
            ProblemKind.TSP, "t", source="synthetic", instance_id="u", options={"formulation": "x"}
        )


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.toml")), ids=lambda p: p.stem)
def test_toml_export_round_trips(path: Path) -> None:
    cfg = load_config(path)
    assert RunConfig.model_validate(tomllib.loads(logic.to_toml(cfg))) == cfg


def test_available_ids() -> None:
    assert registry.available_ids("synthetic") == ["uniform"]
    assert registry.available_ids("nowhere") == []
    for source in ("tsplib", "jsplib", "orlib-cap", "cvrplib"):
        assert isinstance(registry.available_ids(source), list)


@pytest.mark.skipif(not INSTALLED, reason="no solver backend installed")
def test_solve_compare_and_export() -> None:
    cfg = load_config(CONFIG_DIR / "jobshop_synthetic.toml")
    out = logic.solve(cfg)
    assert out.error is None and out.result is not None and out.result.ok
    rows = logic.compare_backends(cfg)
    assert len(rows) == len(INSTALLED)
    assert all(r["valid"] for r in rows)
    assert out.model is not None
    lp = logic.export_model(out.model, "auto").decode().lower()
    assert "makespan" in lp and ("subject to" in lp or "st" in lp)


def test_build_errors_become_messages() -> None:
    cfg = load_config(CONFIG_DIR / "tsp_synthetic.toml")
    cfg = cfg.model_copy(update={"instance": cfg.instance.model_copy(update={"source": "nowhere"})})
    out = logic.solve(cfg)
    assert out.result is None and out.error is not None and "unsupported" in out.error


def test_licence_warning() -> None:
    small = registry.model_from_config(load_config(CONFIG_DIR / "tsp_synthetic.toml"))
    assert logic.licence_warning(small, "gurobi") is None
    cfg = logic.build_config(
        ProblemKind.TSP, "big", source="synthetic", instance_id="uniform", params={"n": 80}
    )
    big = registry.model_from_config(cfg)
    assert logic.licence_warning(big, "gurobi") is not None
    assert logic.licence_warning(big, "highs") is None


# -------------------------------------------------------------------- page
APP = Path(__file__).resolve().parents[1] / "src" / "linear_opt" / "app" / "main.py"


@pytest.fixture
def app():  # type: ignore[no-untyped-def]
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP), default_timeout=120)


@pytest.mark.skipif(not INSTALLED, reason="no solver backend installed")
@pytest.mark.parametrize(
    ("kind", "preset"),
    [
        ("transport", "transport_synthetic"),
        ("facility_location", "facility_synthetic"),
        ("tsp", "tsp_synthetic"),
        ("cvrp", "cvrp_synthetic"),
        ("job_shop", "jobshop_synthetic"),
    ],
)
def test_page_solves_every_problem(app, kind: str, preset: str) -> None:  # type: ignore[no-untyped-def]
    app.run()
    assert not app.exception
    app.selectbox(key="kind").set_value(kind).run()
    app.selectbox(key=f"preset:{kind}").set_value(preset).run()
    app.button(key="solve").click().run()
    assert not app.exception, app.exception
    assert any("re-checked" in s.value for s in app.success)
    assert [m for m in app.metric if m.label == "Objective"]


@pytest.mark.skipif(not INSTALLED, reason="no solver backend installed")
def test_page_compare_variants_and_switching(app) -> None:  # type: ignore[no-untyped-def]
    app.run()
    app.selectbox(key="kind").set_value("job_shop").run()
    app.selectbox(key="preset:job_shop").set_value("jobshop_synthetic").run()
    app.button(key="solve").click().run()
    app.button(key="compare").click().run()
    app.button(key="variants").click().run()
    assert not app.exception
    assert len(app.dataframe) >= 4  # 2 structure tables + comparison + formulations
    app.selectbox(key="kind").set_value("cvrp").run()  # the job-shop result must not leak
    assert not app.exception
    assert any("press **Solve**" in i.value for i in app.info)
