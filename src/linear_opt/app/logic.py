"""UI-independent logic of the dashboard (unit-tested without a browser)."""

from __future__ import annotations

import json
import os
import tempfile
import tomllib
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from linear_opt.core.backends import available_backends
from linear_opt.core.config import ProblemKind, RunConfig
from linear_opt.core.model import OptimizationModel, Result
from linear_opt.problems import registry

PROBLEMS: dict[ProblemKind, str] = {
    ProblemKind.TRANSPORT: "Transport (LP)",
    ProblemKind.FACILITY_LOCATION: "Facility location (MILP)",
    ProblemKind.TSP: "Travelling salesman (MILP)",
    ProblemKind.CVRP: "Vehicle routing (MILP)",
    ProblemKind.JOB_SHOP: "Job-shop scheduling (MILP)",
}

SOURCE_LABELS = {
    "synthetic": "Synthetic (random)",
    "geonames-de": "German cities (GeoNames)",
    "orlib-cap": "OR-Library CAP",
    "tsplib": "TSPLIB",
    "cvrplib": "CVRPLIB",
    "jsplib": "JSPLIB",
}

#: Gurobi's bundled pip licence is limited to about this many variables/constraints.
RESTRICTED_LIMIT = 2000


# ------------------------------------------------------------------- presets
def config_dir() -> Path:
    """Where preset configs live: ``$LINEAR_OPT_CONFIG_DIR``, ``./configs`` or the repo's."""
    env = os.environ.get("LINEAR_OPT_CONFIG_DIR")
    if env:
        return Path(env)
    cwd = Path.cwd() / "configs"
    if cwd.is_dir():
        return cwd
    return Path(__file__).resolve().parents[3] / "configs"


def presets(kind: ProblemKind, folder: Path | None = None) -> dict[str, RunConfig]:
    """Valid preset configs for ``kind``, keyed by file name (invalid files are skipped)."""
    from linear_opt.core.config import ConfigError, load_config

    out: dict[str, RunConfig] = {}
    for path in sorted((folder or config_dir()).glob("*.toml")):
        try:
            cfg = load_config(path)
        except ConfigError:
            continue
        if cfg.problem.kind is kind:
            out[path.stem] = cfg
    return out


def instance_choices(kind: ProblemKind) -> dict[str, list[str]]:
    """Sources with at least one loadable instance, and their ids."""
    out = {}
    for source in registry.SOURCES[kind]:
        ids = registry.available_ids(source)
        if ids:
            out[source] = ids
    return out


# ------------------------------------------------------------------- widgets
@dataclass(frozen=True)
class WidgetSpec:
    """How to render one pydantic field as an input widget."""

    name: str
    kind: Literal["select", "bool", "int", "float", "text"]
    default: Any
    choices: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    help: str = ""


def _bounds(meta: list[Any]) -> tuple[float | None, float | None]:
    lo = hi = None
    for m in meta:
        for attr, is_lo, strict in (
            ("ge", True, False),
            ("gt", True, True),
            ("le", False, False),
            ("lt", False, True),
        ):
            value = getattr(m, attr, None)
            if value is None:
                continue
            step = 1e-9 if strict else 0.0
            if is_lo:
                lo = float(value) + step
            else:
                hi = float(value) - step
    return lo, hi


def widget_specs(model: type[BaseModel] | None) -> list[WidgetSpec]:
    """One :class:`WidgetSpec` per field of a pydantic model (options or params)."""
    if model is None:
        return []
    specs = []
    for name, info in model.model_fields.items():
        ann: Any = info.annotation
        default = info.get_default(call_default_factory=True)
        help_text = info.description or ""
        if typing.get_origin(ann) is Literal:
            specs.append(
                WidgetSpec(name, "select", default, tuple(typing.get_args(ann)), help=help_text)
            )
        elif ann is bool:
            specs.append(WidgetSpec(name, "bool", default, help=help_text))
        elif ann in (int, float):
            lo, hi = _bounds(list(info.metadata))
            if ann is int:
                lo = None if lo is None else float(int(lo) + (1 if lo % 1 else 0))
            kind: Literal["int", "float"] = "int" if ann is int else "float"
            specs.append(WidgetSpec(name, kind, default, minimum=lo, maximum=hi, help=help_text))
        else:
            specs.append(WidgetSpec(name, "text", default, help=help_text))
    return specs


def options_model(kind: ProblemKind) -> type[BaseModel]:
    """Pydantic model of the ``[model]`` table for ``kind``."""
    return registry.MODEL_CLASSES[kind].options_type


def params_model(kind: ProblemKind, source: str) -> type[BaseModel] | None:
    """Pydantic model of ``[instance.params]`` for ``(kind, source)``."""
    return registry.SOURCES[kind].get(source)


# -------------------------------------------------------------------- config
def build_config(
    kind: ProblemKind,
    name: str,
    *,
    source: str,
    instance_id: str,
    seed: int = 0,
    params: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    solver: dict[str, Any] | None = None,
) -> RunConfig:
    """Assemble and validate a run configuration from UI state.

    Raises:
        ValueError: With a readable message if any section is invalid.
    """
    params = dict(params or {})
    options = dict(options or {})
    try:
        model = params_model(kind, source)
        if model is not None:
            model.model_validate(params)
        options_model(kind).model_validate(options)
        return RunConfig.model_validate(
            {
                "problem": {"kind": kind.value, "name": name},
                "instance": {"source": source, "id": instance_id, "seed": seed, "params": params},
                "solver": dict(solver or {}),
                "model": options,
            }
        )
    except ValidationError as exc:
        messages = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise ValueError(messages) from exc


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    return json.dumps(str(value))  # JSON strings are valid TOML basic strings


def to_toml(cfg: RunConfig) -> str:
    """Serialise a config so that ``linopt solve`` reproduces the dashboard run."""
    data = cfg.model_dump(mode="json", exclude_none=True)
    lines: list[str] = ["# Exported from the linear_opt dashboard", ""]

    def table(name: str, values: dict[str, Any]) -> None:
        scalars = {k: v for k, v in values.items() if not isinstance(v, dict)}
        nested = {k: v for k, v in values.items() if isinstance(v, dict)}
        if scalars or not nested:
            lines.append(f"[{name}]")
            lines.extend(f"{k} = {_toml_value(v)}" for k, v in scalars.items())
            lines.append("")
        for key, sub in nested.items():
            if sub:
                table(f"{name}.{key}", sub)

    for section in ("problem", "instance", "solver", "model"):
        if data.get(section) or section != "model":
            table(section, data.get(section, {}))
    text = "\n".join(lines).rstrip() + "\n"
    RunConfig.model_validate(tomllib.loads(text))  # never emit something we cannot read back
    return text


# --------------------------------------------------------------------- solve
@dataclass
class Outcome:
    """Result of a dashboard action, or a readable error."""

    model: OptimizationModel[Any, Any, Any] | None = None
    result: Result[Any] | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def build_model(cfg: RunConfig) -> Outcome:
    """Load the instance and build the model (no solve)."""
    try:
        model = registry.model_from_config(cfg)
        model.build()
    except (ValueError, FileNotFoundError, OSError) as exc:
        return Outcome(error=str(exc))
    return Outcome(model=model)


def solve(cfg: RunConfig, backend: str | None = None) -> Outcome:
    """Build and solve; licence limits and data errors become messages."""
    from linear_opt.solvers import LicenseLimitError

    built = build_model(cfg)
    if built.model is None:
        return built
    try:
        result = built.model.solve(cfg.solver, backend=backend or cfg.solver.backend)
    except LicenseLimitError as exc:
        return Outcome(model=built.model, error=f"{exc}")
    return Outcome(model=built.model, result=result)


def compare_backends(cfg: RunConfig) -> list[dict[str, Any]]:
    """Solve with every installed backend; one row per backend."""
    rows: list[dict[str, Any]] = []
    for name in available_backends():
        out = solve(cfg, name.value)
        if out.result is None:
            rows.append({"backend": name.value, "status": "error", "message": out.error})
            continue
        s = out.result.summary()
        rows.append(
            {
                "backend": s["backend"],
                "status": s["status"],
                "objective": s["objective"],
                "bound": s["bound"],
                "runtime_s": s["runtime_s"],
                "nodes": s["nodes"],
                "valid": out.result.ok,
            }
        )
    return rows


def licence_warning(model: OptimizationModel[Any, Any, Any], backend: str) -> str | None:
    """Warn when a model is likely too large for Gurobi's restricted licence."""
    stats = model.build().stats()
    big = max(stats["variables"], stats["constraints"]) > RESTRICTED_LIMIT
    if big and backend in ("gurobi", "auto"):
        return (
            f"{stats['variables']} variables / {stats['constraints']} constraints exceed the "
            f"restricted Gurobi licence (~{RESTRICTED_LIMIT}). Use HiGHS or a full licence."
        )
    return None


def export_model(
    model: OptimizationModel[Any, Any, Any], backend: str, suffix: str = ".lp"
) -> bytes:
    """The model as an ``.lp``/``.mps`` file, written by the chosen backend."""
    from linear_opt.core.backends import resolve_backend
    from linear_opt.core.config import BackendName
    from linear_opt.solvers import create_backend

    solver = create_backend(resolve_backend(BackendName(backend)))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{model.name}{suffix}"
        solver.write_model(model.build(), str(path))
        return path.read_bytes()
