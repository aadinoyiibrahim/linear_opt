"""Typed run configuration loaded from TOML.

A run is fully described by one TOML file under ``configs/``::

    [problem]
    kind = "facility_location"
    name = "cap41"

    [instance]
    source = "orlib-cap"
    id = "cap41"

    [solver]
    backend = "auto"
    time_limit = 60.0

    [model]                 # free-form, validated by the concrete model
    formulation = "strong"

Validation is strict (``extra="forbid"``): a typo such as ``time_limt`` fails
loudly instead of being silently ignored.
"""

from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ConfigError(ValueError):
    """Raised when a configuration file cannot be read or fails validation."""


class ProblemKind(StrEnum):
    """Problem classes supported by the library."""

    TRANSPORT = "transport"
    FACILITY_LOCATION = "facility_location"
    TSP = "tsp"
    CVRP = "cvrp"
    JOB_SHOP = "job_shop"


class BackendName(StrEnum):
    """Solver backends. ``AUTO`` prefers Gurobi and falls back to HiGHS."""

    AUTO = "auto"
    GUROBI = "gurobi"
    HIGHS = "highs"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProblemSettings(_Strict):
    """Which model to build."""

    kind: ProblemKind
    name: str = Field(min_length=1, description="Human-readable run identifier.")


class InstanceSettings(_Strict):
    """Where the instance data comes from.

    Either a registered dataset (``source`` + ``id``) or a local ``path``.
    """

    source: str | None = Field(default=None, description="Key in data/SOURCES.toml.")
    id: str | None = Field(default=None, description="Instance id within the source.")
    path: Path | None = Field(default=None, description="Local instance file.")
    seed: int = 0
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific options (e.g. number of cities), validated by the loader.",
    )

    @model_validator(mode="after")
    def _exactly_one_origin(self) -> InstanceSettings:
        registered = self.source is not None
        if registered == (self.path is not None):
            raise ValueError("specify exactly one of `source` (+ `id`) or `path`")
        if registered and self.id is None:
            raise ValueError("`id` is required when `source` is given")
        return self


class SolverSettings(_Strict):
    """Solver-independent parameters, mapped onto each backend's native names."""

    backend: BackendName = BackendName.AUTO
    time_limit: float = Field(default=60.0, gt=0, description="Wall-clock seconds.")
    mip_gap: float = Field(default=1e-4, ge=0, le=1, description="Relative MIP gap.")
    threads: int = Field(default=0, ge=0, description="0 = solver default.")
    seed: int = Field(default=0, ge=0)
    verbose: bool = False
    export: Path | None = Field(
        default=None, description="Write the model before solving (.lp / .mps), for debugging."
    )
    gurobi: dict[str, int | float | str] = Field(
        default_factory=dict,
        description="Native Gurobi parameters, e.g. {Cuts = 0, Presolve = 0}; applied last.",
    )
    highs: dict[str, int | float | str | bool] = Field(
        default_factory=dict,
        description='Native HiGHS options, e.g. {presolve = "off"}; applied last.',
    )


class RunConfig(_Strict):
    """Top-level configuration for a single solve."""

    problem: ProblemSettings
    instance: InstanceSettings
    solver: SolverSettings = SolverSettings()
    model: dict[str, Any] = Field(
        default_factory=dict, description="Problem-specific options, validated by the model."
    )


def load_config(path: str | Path) -> RunConfig:
    """Read and validate a TOML run configuration.

    Args:
        path: Path to a ``.toml`` file.

    Returns:
        The validated, immutable configuration.

    Raises:
        ConfigError: If the file is missing, is not valid TOML, or fails validation.
    """
    path = Path(path)
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc

    try:
        return RunConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {path}:\n{exc}") from exc
