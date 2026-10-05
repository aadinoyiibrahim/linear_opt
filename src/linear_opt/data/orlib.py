"""OR-Library capacitated warehouse location instances (Beasley, 1990).

File format (whitespace separated, line breaks are not significant)::

    m n
    capacity_1 fixed_cost_1
    ...                                  (m lines)
    demand_1
    cost_11 cost_21 ... cost_m1          (n blocks: demand, then m costs)
    ...

``cost_ij`` is the cost of serving the **whole** demand of customer ``j`` from
warehouse ``i``; demand may be split across warehouses. Known optimal values
are published in ``capopt.txt``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from linear_opt.core.ir import FloatArray
from linear_opt.data.download import fetch

BASE_URL = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"
#: Instance ids in OR-Library order (cap5x has a single member).
CAP_IDS: tuple[str, ...] = tuple(
    f"cap{group}{k}"
    for group, size in (
        (4, 4),
        (5, 1),
        (6, 4),
        (7, 4),
        (8, 4),
        (9, 4),
        (10, 4),
        (11, 4),
        (12, 4),
        (13, 4),
    )
    for k in range(1, size + 1)
)
OPT_FILE = "capopt.txt"


@dataclass(frozen=True, eq=False)
class CapData:
    """Raw content of one OR-Library ``cap*`` file."""

    name: str
    capacity: FloatArray
    fixed_cost: FloatArray
    demand: FloatArray
    cost: FloatArray  # (m, n): cost of serving all of customer j from warehouse i

    @property
    def shape(self) -> tuple[int, int]:
        """``(m, n)`` = (warehouses, customers)."""
        return int(self.capacity.shape[0]), int(self.demand.shape[0])


def parse_cap(text: str, name: str = "cap") -> CapData:
    """Parse the text of a ``cap*`` file.

    Raises:
        ValueError: If the content does not match the format.
    """
    try:
        tokens = [float(t) for t in text.split()]
    except ValueError as exc:
        raise ValueError(
            f"{name}: non-numeric token ({exc}); only cap41-cap134 are supported"
        ) from exc
    if len(tokens) < 2:
        raise ValueError(f"{name}: file is empty or truncated")
    m, n = int(tokens[0]), int(tokens[1])
    expected = 2 + 2 * m + n * (1 + m)
    if len(tokens) != expected:
        raise ValueError(f"{name}: expected {expected} numbers for m={m}, n={n}, got {len(tokens)}")
    head = np.array(tokens[2 : 2 + 2 * m]).reshape(m, 2)
    body = np.array(tokens[2 + 2 * m :]).reshape(n, 1 + m)
    return CapData(
        name=name,
        capacity=head[:, 0].copy(),
        fixed_cost=head[:, 1].copy(),
        demand=body[:, 0].copy(),
        cost=body[:, 1:].T.copy(),
    )


def parse_capopt(text: str) -> dict[str, float]:
    """Parse ``capopt.txt`` into ``{"cap41": 1040444.375, ...}``."""
    pattern = re.compile(r"^\s*(cap\d+)\s+([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)
    return {name: float(value) for name, value in pattern.findall(text)}


def fixture_dir() -> Path:
    """Directory of packaged OR-Library snapshots."""
    return Path(str(resources.files("linear_opt.data") / "fixtures" / "orlib"))


def _read(filename: str) -> str:
    local = fixture_dir() / filename
    path = local if local.exists() else fetch(BASE_URL + filename)
    return path.read_text(encoding="ascii", errors="strict")


def load_cap(name: str) -> CapData:
    """Load an instance from the packaged snapshot, else download it (cached)."""
    if name not in CAP_IDS:
        raise ValueError(f"unknown instance {name!r}; available: {', '.join(CAP_IDS)}")
    return parse_cap(_read(f"{name}.txt"), name)


def known_optimum(name: str) -> float | None:
    """Published optimum from ``capopt.txt`` (packaged snapshot, else download)."""
    return parse_capopt(_read(OPT_FILE)).get(name)


def snapshot(names: tuple[str, ...] | list[str], target: Path | None = None) -> list[Path]:
    """Download instances plus ``capopt.txt`` and copy them into ``target``."""
    target = target or fixture_dir()
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for filename in (*[f"{n}.txt" for n in names], OPT_FILE):
        text = fetch(BASE_URL + filename).read_text(encoding="ascii")
        if filename != OPT_FILE:
            parse_cap(text, filename)  # refuse to snapshot a malformed file
        (target / filename).write_text(text, encoding="ascii")
        written.append(target / filename)
    return written
