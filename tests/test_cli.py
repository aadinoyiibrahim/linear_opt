from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from linear_opt import __version__
from linear_opt.cli import app
from linear_opt.core.backends import is_available
from linear_opt.core.config import BackendName, load_config
from linear_opt.data import geonames

from .conftest import CONFIG_DIR, INSTALLED
from .test_transport import CITIES

runner = CliRunner()
SYNTHETIC = str(CONFIG_DIR / "transport_synthetic.toml")
needs_solver = pytest.mark.skipif(not INSTALLED, reason="no solver backend installed")


def test_info_prints_version() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_validate_accepts_shipped_configs() -> None:
    paths = [str(p) for p in sorted(CONFIG_DIR.glob("*.toml"))]
    result = runner.invoke(app, ["validate", *paths])
    assert result.exit_code == 0, result.output


def test_validate_fails_on_bad_config(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('[problem]\nkind = "nope"\nname = "x"\n', encoding="utf-8")
    assert runner.invoke(app, ["validate", str(bad)]).exit_code == 1


@needs_solver
def test_solve_writes_json_and_html(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    out_json, out_html = tmp_path / "s.json", tmp_path / "r.html"
    result = runner.invoke(
        app, ["solve", SYNTHETIC, "--json", str(out_json), "--html", str(out_html)]
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(out_json.read_text())
    assert summary["status"] == "optimal" and summary["violations"] == []
    params = load_config(SYNTHETIC).instance.params
    assert summary["model"]["variables"] == params["n_sources"] * params["n_sinks"]
    assert out_html.stat().st_size > 1000


@pytest.mark.skipif(not is_available(BackendName.HIGHS), reason="needs HiGHS")
def test_solve_backend_override() -> None:
    result = runner.invoke(app, ["solve", SYNTHETIC, "--backend", "highs"])
    assert result.exit_code == 0, result.output
    assert "highs" in result.stdout


@needs_solver
def test_compare_reports_agreement() -> None:
    result = runner.invoke(app, ["compare", SYNTHETIC])
    assert result.exit_code == 0, result.output
    if len(INSTALLED) > 1:
        assert "agree" in result.stdout


def test_solve_with_unknown_data_source_exits_cleanly(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[problem]\nkind = "tsp"\nname = "x"\n[instance]\nsource = "nowhere"\nid = "x"\n'
    )
    result = runner.invoke(app, ["solve", str(bad)])
    assert result.exit_code == 2
    assert "unsupported instance source" in result.output


def test_data_cities_writes_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(geonames, "download_cities", lambda country: CITIES)
    out = tmp_path / "cities.csv"
    result = runner.invoke(app, ["data", "cities", "--out", str(out), "--limit", "3"])
    assert result.exit_code == 0, result.output
    assert [c.name for c in geonames.read_cities(out)] == ["Alpha", "Beta", "Gamma"]


FACILITY = str(CONFIG_DIR / "facility_synthetic.toml")


@needs_solver
def test_facility_solve_with_report(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    out = tmp_path / "f.html"
    result = runner.invoke(app, ["solve", FACILITY, "--html", str(out)])
    assert result.exit_code == 0, result.output
    assert out.read_text().count("<section>") == 5


@needs_solver
def test_solve_relaxation_only() -> None:
    result = runner.invoke(app, ["solve", FACILITY, "--relax"])
    assert result.exit_code == 0, result.output
    assert "LP relaxation" in result.stdout


@needs_solver
def test_variants_table_json_and_report(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    out_json, out_html = tmp_path / "v.json", tmp_path / "v.html"
    result = runner.invoke(
        app, ["variants", FACILITY, "--json", str(out_json), "--html", str(out_html)]
    )
    assert result.exit_code == 0, result.output
    rows = json.loads(out_json.read_text())
    assert [r["variant"] for r in rows] == ["strong", "weak + cut", "weak"]
    assert rows[0]["root_gap"] <= rows[2]["root_gap"] + 1e-9
    assert out_html.exists()


def test_variants_requires_a_model_with_variants() -> None:
    assert runner.invoke(app, ["variants", SYNTHETIC]).exit_code == 2


def test_data_orlib_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from linear_opt.data import orlib

    def fake_snapshot(names: list[str], target: Path | None) -> list[Path]:
        assert target is not None
        target.mkdir(parents=True, exist_ok=True)
        (target / "capopt.txt").write_text("cap41 1040444.375\n")
        return [target / "cap41.txt", target / "capopt.txt"]

    monkeypatch.setattr(orlib, "snapshot", fake_snapshot)
    result = runner.invoke(app, ["data", "orlib-cap", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "1,040,444.375" in result.stdout
    assert runner.invoke(app, ["data", "orlib-cap", "cap99"]).exit_code == 2


@needs_solver
@pytest.mark.parametrize("config", ["tsp_synthetic.toml", "cvrp_synthetic.toml"])
def test_routing_solve_with_report(config: str, tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    out = tmp_path / "r.html"
    result = runner.invoke(app, ["solve", str(CONFIG_DIR / config), "--html", str(out)])
    assert result.exit_code == 0, result.output
    assert out.read_text().count("<section>") >= 2


@needs_solver
def test_tsp_variants() -> None:
    result = runner.invoke(app, ["variants", str(CONFIG_DIR / "tsp_synthetic.toml")])
    assert result.exit_code == 0, result.output
    assert "mtz" in result.stdout


def test_data_tsplib_and_cvrplib(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from linear_opt.data import tsplib
    from linear_opt.data.download import DownloadError

    monkeypatch.setattr(
        tsplib, "snapshot_tsplib", lambda name, out, from_dir=None: [tmp_path / f"{name}.tsp"]
    )
    result = runner.invoke(app, ["data", "tsplib", "toy", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "toy.tsp" in result.stdout

    def failing(*args: object, **kwargs: object) -> list[Path]:
        raise DownloadError("blocked")

    monkeypatch.setattr(tsplib, "snapshot_cvrplib", failing)
    result = runner.invoke(app, ["data", "cvrplib", "A-n32-k5"])
    assert result.exit_code == 1
    assert "--from-dir" in result.output


@needs_solver
def test_jobshop_solve_and_variants(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    config = str(CONFIG_DIR / "jobshop_synthetic.toml")
    out = tmp_path / "g.html"
    result = runner.invoke(app, ["solve", config, "--html", str(out)])
    assert result.exit_code == 0, result.output
    assert out.read_text().count("<section>") >= 3
    rows_json = tmp_path / "v.json"
    result = runner.invoke(app, ["variants", config, "--json", str(rows_json)])
    assert result.exit_code == 0, result.output
    names = [r["variant"] for r in json.loads(rows_json.read_text())]
    assert names == ["disjunctive (tight M)", "disjunctive (naive M)", "time-indexed"]


def test_data_jsplib(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from linear_opt.data import jsplib

    monkeypatch.setattr(jsplib, "snapshot", lambda names, out, orlib_file=None: [tmp_path / "ft06"])
    result = runner.invoke(app, ["data", "jsplib", "ft06", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
