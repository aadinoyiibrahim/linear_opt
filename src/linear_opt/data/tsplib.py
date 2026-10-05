"""TSPLIB 95 and CVRPLIB instances (Reinelt 1991; Augerat et al. 1995).

Both libraries use the TSPLIB file format: ``KEY : VALUE`` header lines
followed by sections (``NODE_COORD_SECTION``, ``EDGE_WEIGHT_SECTION``,
``DEMAND_SECTION``, ``DEPOT_SECTION``, ``TOUR_SECTION``). Distances follow the
TSPLIB specification exactly - in particular they are **rounded to integers**
(``EUC_2D`` uses nearest-integer rounding, ``ATT`` a pseudo-Euclidean rule,
``GEO`` great circles on an idealised sphere) - otherwise published optima are
not reproducible.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from urllib.parse import urljoin

import numpy as np

from linear_opt.core.ir import FloatArray
from linear_opt.data.download import fetch

#: TSPLIB sources, tried in order: (base URL, files are gzipped?). The original
#: Heidelberg server is often slow or unreachable; the GitHub mirror (MIT
#: licensed, pdrozdowski/TSPLib.Net) carries the same files uncompressed.
TSPLIB_MIRRORS: tuple[tuple[str, bool], ...] = (
    ("https://raw.githubusercontent.com/pdrozdowski/TSPLib.Net/master/TSPLIB95/tsp/", False),
    ("http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/", True),
)
#: CVRPLIB site. Downloads use numeric ids (``download/instance/<id>``,
#: ``download/bks/<id>``) that are looked up on the instance list page.
CVRPLIB_URL = "https://galgos.inf.puc-rio.br/cvrplib/en/"

#: Optimal tour lengths published with TSPLIB 95 (STSP table). The test suite
#: cross-checks them against the downloadable optimal tours where available.
TSP_OPTIMA: dict[str, int] = {
    "burma14": 3323,
    "ulysses16": 6859,
    "gr17": 2085,
    "ulysses22": 7013,
    "gr24": 1272,
    "bayg29": 1610,
    "bays29": 2020,
    "dantzig42": 699,
    "att48": 10628,
    "eil51": 426,
    "berlin52": 7542,
    "st70": 675,
    "eil76": 538,
    "kroA100": 21282,
}

_SECTIONS = {
    "NODE_COORD_SECTION",
    "EDGE_WEIGHT_SECTION",
    "DISPLAY_DATA_SECTION",
    "DEMAND_SECTION",
    "DEPOT_SECTION",
    "TOUR_SECTION",
    "FIXED_EDGES_SECTION",
}


@dataclass(frozen=True, eq=False)
class TsplibProblem:
    """A parsed TSPLIB / CVRPLIB file (nodes are 0-based)."""

    name: str
    kind: str  # "TSP" or "CVRP"
    distances: FloatArray  # (n, n), integer-valued per the TSPLIB rules
    coords: FloatArray | None = None  # (n, 2) node or display coordinates
    edge_weight_type: str = ""
    comment: str = ""
    demand: FloatArray | None = None  # CVRP
    capacity: float | None = None  # CVRP
    depot: int | None = None  # CVRP
    vehicles: int | None = None  # CVRP, from VEHICLES or the "-kN" name suffix

    @property
    def dimension(self) -> int:
        """Number of nodes."""
        return int(self.distances.shape[0])


def _split(text: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    header: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        key = line.split(":")[0].strip().upper() if ":" in line else line.split()[0].upper()
        if line.upper() == "EOF":
            break
        if key in _SECTIONS:
            current = key
            sections[current] = []
            rest = line[len(key) :].lstrip(" :")
            if rest:
                sections[current].extend(rest.split())
            continue
        if current is None or (":" in line and re.match(r"^[A-Z_]+\s*:", line)):
            if ":" in line:
                k, v = line.split(":", 1)
                header[k.strip().upper()] = v.strip()
                current = None
            continue
        sections[current].extend(line.split())
    return header, sections


def _nint(x: FloatArray) -> FloatArray:
    return np.floor(x + 0.5)


def euclidean_distances(coords: FloatArray, rule: str = "EUC_2D") -> FloatArray:
    """TSPLIB distances for planar coordinates (``EUC_2D``, ``CEIL_2D``, ``ATT``)."""
    diff = coords[:, None, :] - coords[None, :, :]
    sq = (diff**2).sum(axis=2)
    if rule == "EUC_2D":
        return _nint(np.sqrt(sq))
    if rule == "CEIL_2D":
        return np.ceil(np.sqrt(sq) - 1e-12)
    if rule == "ATT":
        r = np.sqrt(sq / 10.0)
        t = _nint(r)
        return np.where(t < r, t + 1.0, t)
    raise ValueError(f"unsupported planar rule {rule}")


def geo_distances(coords: FloatArray) -> FloatArray:
    """TSPLIB ``GEO`` distances; coordinates are ``DDD.MM`` (degrees.minutes)."""
    pi, radius = 3.141592, 6378.388
    deg = np.trunc(coords)
    rad = pi * (deg + 5.0 * (coords - deg) / 3.0) / 180.0
    lat, lon = rad[:, 0], rad[:, 1]
    q1 = np.cos(lon[:, None] - lon[None, :])
    q2 = np.cos(lat[:, None] - lat[None, :])
    q3 = np.cos(lat[:, None] + lat[None, :])
    arg = np.clip(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3), -1.0, 1.0)
    d = np.floor(radius * np.arccos(arg) + 1.0)
    np.fill_diagonal(d, 0.0)
    return d


def explicit_distances(values: list[float], n: int, fmt: str) -> FloatArray:
    """Matrix from an ``EDGE_WEIGHT_SECTION`` in any standard format."""
    fmt = {
        "UPPER_COL": "LOWER_ROW",
        "LOWER_COL": "UPPER_ROW",
        "UPPER_DIAG_COL": "LOWER_DIAG_ROW",
        "LOWER_DIAG_COL": "UPPER_DIAG_ROW",
    }.get(fmt, fmt)
    d = np.zeros((n, n))
    if fmt == "FULL_MATRIX":
        if len(values) != n * n:
            raise ValueError(f"FULL_MATRIX needs {n * n} values, got {len(values)}")
        return np.asarray(values, dtype=np.float64).reshape(n, n)
    pairs = {
        "UPPER_ROW": [(i, j) for i in range(n) for j in range(i + 1, n)],
        "LOWER_ROW": [(i, j) for i in range(n) for j in range(i)],
        "UPPER_DIAG_ROW": [(i, j) for i in range(n) for j in range(i, n)],
        "LOWER_DIAG_ROW": [(i, j) for i in range(n) for j in range(i + 1)],
    }.get(fmt)
    if pairs is None:
        raise ValueError(f"unsupported EDGE_WEIGHT_FORMAT {fmt}")
    if len(values) != len(pairs):
        raise ValueError(f"{fmt} needs {len(pairs)} values, got {len(values)}")
    rows, cols = np.array(pairs).T
    d[rows, cols] = values
    d[cols, rows] = values
    return d


def parse_tsplib(text: str) -> TsplibProblem:
    """Parse a ``.tsp`` or ``.vrp`` file.

    Raises:
        ValueError: On unsupported or inconsistent content.
    """
    header, sections = _split(text)
    name = header.get("NAME", "unnamed")
    kind = header.get("TYPE", "TSP").split()[0].upper()
    if kind not in {"TSP", "CVRP"}:
        raise ValueError(f"{name}: unsupported TYPE {kind} (only symmetric TSP and CVRP)")
    n = int(header["DIMENSION"])
    rule = header.get("EDGE_WEIGHT_TYPE", "").upper()

    def node_table(section: str) -> FloatArray | None:
        if section not in sections:
            return None
        vals = np.asarray(sections[section], dtype=np.float64)
        if vals.size != 3 * n:
            raise ValueError(f"{name}: {section} needs {3 * n} numbers, got {vals.size}")
        table = vals.reshape(n, 3)
        order = np.argsort(table[:, 0])  # node ids may be unsorted
        return table[order, 1:]

    coords = node_table("NODE_COORD_SECTION")
    display = node_table("DISPLAY_DATA_SECTION")
    if rule in {"EUC_2D", "CEIL_2D", "ATT"}:
        if coords is None:
            raise ValueError(f"{name}: {rule} needs NODE_COORD_SECTION")
        dist = euclidean_distances(coords, rule)
    elif rule == "GEO":
        if coords is None:
            raise ValueError(f"{name}: GEO needs NODE_COORD_SECTION")
        dist = geo_distances(coords)
    elif rule == "EXPLICIT":
        fmt = header.get("EDGE_WEIGHT_FORMAT", "").upper()
        values = [float(v) for v in sections.get("EDGE_WEIGHT_SECTION", [])]
        dist = explicit_distances(values, n, fmt)
    else:
        raise ValueError(f"{name}: unsupported EDGE_WEIGHT_TYPE {rule!r}")
    np.fill_diagonal(dist, 0.0)

    demand = capacity = depot = vehicles = None
    if kind == "CVRP":
        capacity = float(header["CAPACITY"])
        table = np.asarray(sections.get("DEMAND_SECTION", []), dtype=np.float64).reshape(-1, 2)
        if table.shape[0] != n:
            raise ValueError(f"{name}: DEMAND_SECTION has {table.shape[0]} rows, expected {n}")
        demand = table[np.argsort(table[:, 0]), 1]
        depots = [int(v) for v in sections.get("DEPOT_SECTION", ["1"]) if int(v) > 0]
        if len(depots) != 1:
            raise ValueError(f"{name}: exactly one depot supported")
        depot = depots[0] - 1
        if "VEHICLES" in header:
            vehicles = int(header["VEHICLES"])
        elif match := re.search(r"-k(\d+)", name):
            vehicles = int(match.group(1))
    return TsplibProblem(
        name=name,
        kind=kind,
        distances=dist,
        coords=coords if coords is not None else display,
        edge_weight_type=rule,
        comment=header.get("COMMENT", ""),
        demand=demand,
        capacity=capacity,
        depot=depot,
        vehicles=vehicles,
    )


def parse_tour(text: str) -> list[int]:
    """0-based node sequence from a ``.tour`` / ``.opt.tour`` file."""
    _, sections = _split(text)
    nodes = []
    for token in sections.get("TOUR_SECTION", []):
        value = int(token)
        if value == -1:
            break
        nodes.append(value - 1)
    if not nodes:
        raise ValueError("no TOUR_SECTION found")
    return nodes


def parse_cvrp_solution(text: str) -> tuple[list[list[int]], float | None]:
    """Routes (customer node indices, depot = 0 excluded) and cost from a CVRPLIB ``.sol``.

    CVRPLIB numbers customers 1..n-1 with the depot as 0, which coincides with
    0-based node indices of the ``.vrp`` file when the depot is node 1.
    """
    routes = [
        [int(v) for v in m.group(1).split()]
        for m in re.finditer(r"^\s*Route\s*#\s*\d+\s*:\s*(.*)$", text, re.MULTILINE | re.IGNORECASE)
    ]
    cost = re.search(r"^\s*Cost\s+([0-9.]+)", text, re.MULTILINE | re.IGNORECASE)
    return routes, float(cost.group(1)) if cost else None


def tour_length(dist: FloatArray, tour: Iterable[int]) -> float:
    """Length of the closed tour."""
    t = np.asarray(list(tour), dtype=np.int64)
    return float(dist[t, np.roll(t, -1)].sum())


# ---------------------------------------------------------------- loading
def fixture_dir(library: str) -> Path:
    """Directory of packaged snapshots: ``tsplib`` or ``cvrplib``."""
    return Path(str(resources.files("linear_opt.data") / "fixtures" / library))


def fetch_tsplib_file(filename: str) -> str:
    """Text of a TSPLIB file from the first mirror that serves it.

    Raises:
        DownloadError: If no mirror has the file (the last error is reported).
    """
    from linear_opt.data.download import DownloadError

    errors = []
    for base, gzipped in TSPLIB_MIRRORS:
        url = f"{base}{filename}{'.gz' if gzipped else ''}"
        try:
            path = fetch(url, timeout=20)
        except DownloadError as exc:
            errors.append(str(exc))
            continue
        data = path.read_bytes()
        return (gzip.decompress(data) if gzipped else data).decode("ascii")
    raise DownloadError(f"{filename}: no mirror succeeded:\n  " + "\n  ".join(errors))


def snapshot_tsplib(
    name: str, target: Path | None = None, *, from_dir: Path | None = None
) -> list[Path]:
    """Store ``<name>.tsp`` (and ``<name>.opt.tour`` if published) in ``target``."""
    from linear_opt.data.download import DownloadError

    target = target or fixture_dir("tsplib")
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, required in ((".tsp", True), (".opt.tour", False)):
        filename = f"{name}{suffix}"
        try:
            if from_dir is not None:
                source = from_dir / filename
                if not source.exists():
                    raise DownloadError(f"{source} not found")
                text = source.read_text(encoding="ascii")
            else:
                text = fetch_tsplib_file(filename)
        except DownloadError:
            if required:
                raise
            continue
        if suffix == ".tsp":
            parse_tsplib(text)  # refuse malformed files
        else:
            parse_tour(text)
        (target / filename).write_text(text, encoding="ascii")
        written.append(target / filename)
    return written


def cvrplib_links(name: str, page_html: str) -> tuple[str, str]:
    """Instance and best-known-solution URLs for ``name`` from the instance list HTML."""
    anchor = re.compile(
        r'href="(?P<href>[^"]*/download/instance/(?P<id>\d+))"[^>]*>\s*(?:<[^>]+>\s*)*'
        + re.escape(name)
        + r"\s*<",
        re.IGNORECASE,
    )
    match = anchor.search(page_html)
    if match is None:
        raise ValueError(f"{name!r} not found on the CVRPLIB instance list")
    href = match.group("href")
    return href, href.replace("/download/instance/", "/download/bks/")


def snapshot_cvrplib(
    name: str,
    target: Path | None = None,
    *,
    base_url: str = CVRPLIB_URL,
    from_dir: Path | None = None,
) -> list[Path]:
    """Fetch ``<name>.vrp`` and ``<name>.sol`` (from CVRPLIB or a local directory)."""
    target = target or fixture_dir("cvrplib")
    target.mkdir(parents=True, exist_ok=True)
    links: tuple[str, str] | None = None
    if from_dir is None:
        listing_url = f"{base_url.rstrip('/')}/instances"
        listing = fetch(listing_url, "cvrplib_instances.html", force=True)
        found = cvrplib_links(name, listing.read_text(encoding="utf-8", errors="replace"))
        # The site uses root-relative links ("/cvrplib/en/download/instance/4").
        links = (urljoin(listing_url, found[0]), urljoin(listing_url, found[1]))
    written = []
    for k, suffix in enumerate((".vrp", ".sol")):
        if links is None:
            assert from_dir is not None
            source = from_dir / f"{name}{suffix}"
            if not source.exists():
                raise FileNotFoundError(f"{source} not found")
        else:
            source = fetch(links[k], f"{name}{suffix}", force=True)
        text = source.read_text(encoding="utf-8")
        if suffix == ".vrp":
            parse_tsplib(text)
        (target / f"{name}{suffix}").write_text(text, encoding="utf-8")
        written.append(target / f"{name}{suffix}")
    return written


def load_tsplib(name: str) -> tuple[TsplibProblem, float | None, list[int] | None]:
    """Load a TSPLIB instance: problem, published optimum, optimal tour (if any)."""
    local = fixture_dir("tsplib") / f"{name}.tsp"
    text = local.read_text(encoding="ascii") if local.exists() else fetch_tsplib_file(f"{name}.tsp")
    problem = parse_tsplib(text)
    tour_file = fixture_dir("tsplib") / f"{name}.opt.tour"
    tour = parse_tour(tour_file.read_text(encoding="ascii")) if tour_file.exists() else None
    optimum = TSP_OPTIMA.get(name)
    if optimum is None and tour is not None:
        optimum = int(tour_length(problem.distances, tour))
    return problem, None if optimum is None else float(optimum), tour


def load_cvrplib(name: str) -> tuple[TsplibProblem, float | None, list[list[int]] | None]:
    """Load a CVRPLIB instance from the packaged snapshot: problem, optimum, routes."""
    local = fixture_dir("cvrplib") / f"{name}.vrp"
    if not local.exists():
        raise FileNotFoundError(
            f"{local.name} not found in {local.parent}; run `uv run linopt data cvrplib {name}`"
        )
    problem = parse_tsplib(local.read_text(encoding="utf-8"))
    sol_file = fixture_dir("cvrplib") / f"{name}.sol"
    routes: list[list[int]] | None = None
    cost: float | None = None
    if sol_file.exists():
        routes, cost = parse_cvrp_solution(sol_file.read_text(encoding="utf-8"))
    if cost is None and (m := re.search(r"Optimal value:\s*(\d+)", problem.comment)):
        cost = float(m.group(1))
    return problem, cost, routes
