"""Job-shop benchmark instances (JSPLIB, OR-Library).

Standard format (JSPLIB ``instances/<name>``; ``#`` lines are comments)::

    n m
    machine time machine time ...     (one line per job, operations in order)

Machines are 0-based. OR-Library's ``jobshop1.txt`` holds many instances in the
same format, each introduced by an ``instance <name>`` line.

Optimal makespans (or lower/upper bounds for open instances) come from JSPLIB's
``instances.json`` - a source file, not hand-typed numbers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import numpy.typing as npt

from linear_opt.core.ir import FloatArray
from linear_opt.data.download import fetch

JSPLIB_URL = "https://raw.githubusercontent.com/tamy0612/JSPLIB/master/"
META_FILE = "instances.json"


@dataclass(frozen=True, eq=False)
class JobShopData:
    """Routing (machine per operation) and processing times of each job."""

    name: str
    machines: npt.NDArray[np.int64]  # (n_jobs, n_ops)
    durations: FloatArray  # (n_jobs, n_ops)


@dataclass(frozen=True)
class Reference:
    """Published optimum, or bounds if the instance is open."""

    optimum: float | None
    lower: float | None = None
    upper: float | None = None


def _numbers(lines: list[str]) -> list[int]:
    return [int(tok) for line in lines for tok in line.split()]


def parse_jsp(text: str, name: str = "jsp") -> JobShopData:
    """Parse one instance in the standard format.

    Raises:
        ValueError: If the numbers do not match ``n m`` and ``n`` rows of ``m`` pairs.
    """
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{name}: empty instance")
    n, m = (int(v) for v in lines[0].split()[:2])
    values = _numbers(lines[1:])
    if len(values) != 2 * n * m:
        raise ValueError(f"{name}: expected {2 * n * m} numbers for {n}x{m}, got {len(values)}")
    table = np.asarray(values, dtype=np.int64).reshape(n, m, 2)
    machines, durations = table[:, :, 0], table[:, :, 1].astype(np.float64)
    if machines.min() < 0 or machines.max() >= m:
        raise ValueError(f"{name}: machine index outside 0..{m - 1}")
    return JobShopData(name, machines, durations)


def parse_orlib_jobshop(text: str) -> dict[str, JobShopData]:
    """Split OR-Library ``jobshop1.txt`` into instances (``instance <name>`` headers)."""
    blocks = re.split(r"^\s*instance\s+(\S+)\s*$", text, flags=re.MULTILINE)
    out: dict[str, JobShopData] = {}
    for name, body in zip(blocks[1::2], blocks[2::2], strict=True):
        lines = [ln for ln in body.splitlines() if re.fullmatch(r"[\d\s]+", ln.strip() or "x")]
        out[name] = parse_jsp("\n".join(lines), name)
    return out


def fixture_dir() -> Path:
    """Directory of packaged snapshots."""
    return Path(str(resources.files("linear_opt.data") / "fixtures" / "jsplib"))


def _metadata() -> list[dict[str, object]]:
    local = fixture_dir() / META_FILE
    path = local if local.exists() else fetch(JSPLIB_URL + META_FILE, "jsplib_instances.json")
    data: list[dict[str, object]] = json.loads(path.read_text(encoding="utf-8"))
    return data


def reference(name: str) -> Reference | None:
    """Published optimum / bounds from ``instances.json``."""
    for entry in _metadata():
        if entry.get("name") == name:
            bounds = entry.get("bounds") or {}
            assert isinstance(bounds, dict)
            opt = entry.get("optimum")
            return Reference(
                float(opt) if isinstance(opt, int | float) else None,
                float(bounds["lower"]) if "lower" in bounds else None,
                float(bounds["upper"]) if "upper" in bounds else None,
            )
    return None


def load_jsp(name: str) -> tuple[JobShopData, Reference | None]:
    """Instance (packaged snapshot, else download) and its reference value."""
    local = fixture_dir() / name
    if local.exists():
        text = local.read_text(encoding="utf-8")
    else:
        text = fetch(f"{JSPLIB_URL}instances/{name}", f"jsplib_{name}").read_text(encoding="utf-8")
    return parse_jsp(text, name), reference(name)


def snapshot(
    names: list[str], target: Path | None = None, *, orlib_file: Path | None = None
) -> list[Path]:
    """Store instances (+ ``instances.json``) from JSPLIB, or from a local ``jobshop1.txt``."""
    target = target or fixture_dir()
    target.mkdir(parents=True, exist_ok=True)
    written = []
    if orlib_file is not None:
        found = parse_orlib_jobshop(orlib_file.read_text(encoding="utf-8"))
        missing = sorted(set(names) - set(found))
        if missing:
            raise ValueError(f"not in {orlib_file.name}: {', '.join(missing)}")
        for name in names:
            data = found[name]
            rows = [
                " ".join(f"{mc} {int(d)}" for mc, d in zip(mrow, drow, strict=True))
                for mrow, drow in zip(data.machines, data.durations, strict=True)
            ]
            text = f"# instance {name} (from OR-Library jobshop1.txt)\n"
            text += f"{data.machines.shape[0]} {data.machines.shape[1]}\n" + "\n".join(rows) + "\n"
            (target / name).write_text(text, encoding="utf-8")
            written.append(target / name)
        return written
    for name in names:
        text = fetch(f"{JSPLIB_URL}instances/{name}", f"jsplib_{name}", force=True).read_text(
            encoding="utf-8"
        )
        parse_jsp(text, name)  # refuse malformed files
        (target / name).write_text(text, encoding="utf-8")
        written.append(target / name)
    meta = fetch(JSPLIB_URL + META_FILE, "jsplib_instances.json", force=True)
    (target / META_FILE).write_text(meta.read_text(encoding="utf-8"), encoding="utf-8")
    written.append(target / META_FILE)
    return written
