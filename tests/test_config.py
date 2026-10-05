from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from linear_opt.core.config import (
    BackendName,
    ConfigError,
    ProblemKind,
    RunConfig,
    SolverSettings,
    load_config,
)

from .conftest import CONFIG_DIR, WriteToml

MINIMAL = """
[problem]
kind = "tsp"
name = "berlin52"

[instance]
source = "tsplib"
id = "berlin52"
"""


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.toml")), ids=lambda p: p.stem)
def test_shipped_configs_are_valid(path: Path) -> None:
    cfg = load_config(path)
    assert isinstance(cfg, RunConfig)


def test_shipped_configs_cover_every_problem_kind() -> None:
    kinds = {load_config(p).problem.kind for p in CONFIG_DIR.glob("*.toml")}
    assert kinds == set(ProblemKind)


def test_defaults_are_applied(write_toml: WriteToml) -> None:
    cfg = load_config(write_toml(MINIMAL))
    assert cfg.solver.backend is BackendName.AUTO
    assert cfg.solver.time_limit == 60.0
    assert cfg.model == {}


def test_config_is_immutable(write_toml: WriteToml) -> None:
    cfg = load_config(write_toml(MINIMAL))
    with pytest.raises(ValidationError):
        cfg.solver.time_limit = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("extra", "fragment"),
    [
        ("[solver]\ntime_limt = 5\n", "time_limt"),  # typo -> rejected, not ignored
        ("[solver]\ntime_limit = -1\n", "time_limit"),
        ("[solver]\nmip_gap = 2.0\n", "mip_gap"),
        ('[solver]\nbackend = "cplex"\n', "backend"),
    ],
)
def test_invalid_solver_settings_are_rejected(
    write_toml: WriteToml, extra: str, fragment: str
) -> None:
    with pytest.raises(ConfigError, match=fragment):
        load_config(write_toml(MINIMAL + extra))


def test_unknown_problem_kind_is_rejected(write_toml: WriteToml) -> None:
    with pytest.raises(ConfigError, match="kind"):
        load_config(write_toml(MINIMAL.replace('"tsp"', '"knapsack"')))


@pytest.mark.parametrize(
    "instance",
    [
        '[instance]\nsource = "tsplib"\n',  # source without id
        '[instance]\nsource = "tsplib"\nid = "x"\npath = "a.tsp"\n',  # both origins
        "[instance]\nseed = 1\n",  # neither origin
    ],
)
def test_instance_origin_must_be_unambiguous(write_toml: WriteToml, instance: str) -> None:
    text = '[problem]\nkind = "tsp"\nname = "t"\n' + instance
    with pytest.raises(ConfigError, match="instance"):
        load_config(write_toml(text))


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_malformed_toml_raises_config_error(write_toml: WriteToml) -> None:
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(write_toml("[problem\nkind = "))


@given(
    time_limit=st.floats(min_value=1e-3, max_value=1e6, allow_nan=False),
    mip_gap=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    threads=st.integers(min_value=0, max_value=256),
)
def test_solver_settings_accept_valid_ranges(
    time_limit: float, mip_gap: float, threads: int
) -> None:
    s = SolverSettings(time_limit=time_limit, mip_gap=mip_gap, threads=threads)
    assert (s.time_limit, s.mip_gap, s.threads) == (time_limit, mip_gap, threads)
