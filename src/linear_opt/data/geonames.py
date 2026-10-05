"""German (or any country's) cities from GeoNames ``cities15000``.

GeoNames publishes a tab-separated dump of all places with population > 15 000
(CC BY 4.0). We keep only *populated places* and drop city sections
(``PPLX``), historical (``PPLH``) and abandoned (``PPLQ``) entries, which would
otherwise double-count large cities such as Berlin.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from linear_opt.data.download import fetch

CITIES_URL = "https://download.geonames.org/export/dump/cities15000.zip"
POPULATED_PLACES = frozenset({"PPL", "PPLA", "PPLA2", "PPLA3", "PPLA4", "PPLC", "PPLG"})
FIXTURE_NAME = "de_cities.csv"

# Column positions in the GeoNames "geoname" table (see the dump's readme.txt).
_ID, _NAME, _LAT, _LON, _FCLASS, _FCODE, _COUNTRY, _POP = 0, 1, 4, 5, 6, 7, 8, 14


@dataclass(frozen=True)
class City:
    """A populated place."""

    geonameid: int
    name: str
    lat: float
    lon: float
    population: int


def parse_geonames(lines: Iterable[str], country: str = "DE") -> list[City]:
    """Parse GeoNames rows, keeping populated places of ``country``, largest first."""
    cities: list[City] = []
    for line in lines:
        cols = line.rstrip("\n").split("\t")
        if len(cols) < 15 or cols[_COUNTRY] != country or cols[_FCLASS] != "P":
            continue
        if cols[_FCODE] not in POPULATED_PLACES:
            continue
        population = int(cols[_POP] or 0)
        if population <= 0:
            continue
        cities.append(
            City(int(cols[_ID]), cols[_NAME], float(cols[_LAT]), float(cols[_LON]), population)
        )
    cities.sort(key=lambda c: (-c.population, c.name))
    return cities


def download_cities(country: str = "DE", *, force: bool = False) -> list[City]:
    """Download ``cities15000.zip`` (cached) and return the country's cities.

    The upstream file is regenerated daily, so it cannot be pinned by checksum;
    use :func:`write_cities` to freeze a snapshot for reproducible runs.
    """
    archive = fetch(CITIES_URL, force=force)
    with zipfile.ZipFile(archive) as zf, zf.open("cities15000.txt") as raw:
        return parse_geonames(io.TextIOWrapper(raw, encoding="utf-8"), country)


def write_cities(cities: Sequence[City], path: Path) -> None:
    """Write cities to CSV (the packaged-fixture format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        fh.write("# Source: GeoNames cities15000 (geonames.org), CC BY 4.0\n")
        writer = csv.writer(fh)
        writer.writerow(["geonameid", "name", "lat", "lon", "population"])
        for c in cities:
            writer.writerow([c.geonameid, c.name, f"{c.lat:.5f}", f"{c.lon:.5f}", c.population])


def read_cities(path: Path) -> list[City]:
    """Read cities written by :func:`write_cities`."""
    with path.open(encoding="utf-8") as fh:
        rows = csv.DictReader(line for line in fh if not line.startswith("#"))
        return [
            City(
                int(r["geonameid"]),
                r["name"],
                float(r["lat"]),
                float(r["lon"]),
                int(r["population"]),
            )
            for r in rows
        ]


def fixture_path() -> Path:
    """Location of the packaged snapshot (may not exist until generated)."""
    return Path(str(resources.files("linear_opt.data") / "fixtures" / FIXTURE_NAME))


def load_fixture() -> list[City]:
    """Load the packaged German-cities snapshot.

    Raises:
        FileNotFoundError: With instructions if the snapshot has not been generated.
    """
    path = fixture_path()
    if not path.exists():
        raise FileNotFoundError(
            f"city snapshot not found at {path}. Create it once with\n"
            "    uv run linopt data cities\n"
            'or set `id = "live"` in the config to download on the fly.'
        )
    return read_cities(path)
