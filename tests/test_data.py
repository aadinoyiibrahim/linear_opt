from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from linear_opt.data import distances, download, geonames
from linear_opt.data.download import DownloadError, fetch


def _row(gid: int, name: str, fclass: str, fcode: str, country: str, pop: int) -> str:
    cols = [""] * 19
    cols[0], cols[1], cols[4], cols[5] = str(gid), name, "51.0", "10.0"
    cols[6], cols[7], cols[8], cols[14] = fclass, fcode, country, str(pop)
    return "\t".join(cols) + "\n"


def test_parse_geonames_keeps_only_populated_places_of_country() -> None:
    lines = [
        _row(1, "Smallville", "P", "PPL", "DE", 20_000),
        _row(2, "Capital", "P", "PPLC", "DE", 3_000_000),
        _row(3, "Capital-Ost", "P", "PPLX", "DE", 400_000),  # city section -> dropped
        _row(4, "Elsewhere", "P", "PPL", "AT", 90_000),  # other country -> dropped
        _row(5, "Old Town", "P", "PPLH", "DE", 50_000),  # historical -> dropped
        _row(6, "Mountain", "T", "MT", "DE", 0),  # not a place -> dropped
        "malformed\tline\n",
    ]
    cities = geonames.parse_geonames(lines, "DE")
    assert [c.name for c in cities] == ["Capital", "Smallville"]  # largest first
    assert cities[0].population == 3_000_000 and cities[0].lat == 51.0


def test_city_csv_roundtrip(tmp_path: Path) -> None:
    cities = [geonames.City(1, "Köln", 50.93333, 6.95, 1_000_000)]
    path = tmp_path / "c.csv"
    geonames.write_cities(cities, path)
    assert path.read_text(encoding="utf-8").startswith("# Source: GeoNames")
    assert geonames.read_cities(path) == cities


def test_load_fixture_missing_gives_instructions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(geonames, "fixture_path", lambda: tmp_path / "none.csv")
    with pytest.raises(FileNotFoundError, match="linopt data cities"):
        geonames.load_fixture()


def test_fetch_caches_and_verifies_checksum(tmp_path: Path) -> None:
    src = tmp_path / "payload.txt"
    src.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    first = fetch(src.as_uri(), "payload.txt", sha256=digest)
    assert first.read_bytes() == b"hello" and first.parent == download.cache_dir()
    src.write_bytes(b"changed")  # cached copy is reused
    assert fetch(src.as_uri(), "payload.txt", sha256=digest).read_bytes() == b"hello"


def test_fetch_rejects_checksum_mismatch(tmp_path: Path) -> None:
    src = tmp_path / "payload.txt"
    src.write_bytes(b"hello")
    with pytest.raises(DownloadError, match="checksum"):
        fetch(src.as_uri(), "bad.txt", sha256="0" * 64)
    assert not (download.cache_dir() / "bad.txt").exists()
    assert not (download.cache_dir() / "bad.txt.part").exists()


def test_fetch_wraps_network_errors(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="could not download"):
        fetch((tmp_path / "does-not-exist").as_uri(), "x.bin")


def test_haversine_berlin_munich() -> None:
    berlin, munich = np.array([[52.5200, 13.4050]]), np.array([[48.1372, 11.5756]])
    d = distances.haversine_matrix(berlin, munich)
    assert d.shape == (1, 1)
    assert d[0, 0] == pytest.approx(504, abs=3)  # great-circle distance, km
    assert distances.haversine_matrix(berlin, berlin)[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_road_distances_parse_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake(url: str, timeout: float) -> Any:
        calls.append(url)
        return {"code": "Ok", "distances": [[585_000.0, 290_000.0]]}

    monkeypatch.setattr(distances, "_http_get_json", fake)
    a = np.array([[52.52, 13.405]])
    b = np.array([[48.137, 11.576], [53.55, 9.99]])
    km = distances.road_distance_matrix(a, b)
    np.testing.assert_allclose(km, [[585.0, 290.0]])
    assert "sources=0&destinations=1;2" in calls[0]
    assert "13.40500,52.52000" in calls[0]  # OSRM expects lon,lat
    distances.road_distance_matrix(a, b)
    assert len(calls) == 1  # second call served from cache


@pytest.mark.parametrize(
    "payload",
    [{"code": "Ok", "distances": [[None]]}, {"code": "InvalidQuery", "message": "bad"}],
)
def test_road_distances_errors(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    monkeypatch.setattr(distances, "_http_get_json", lambda url, timeout: payload)
    with pytest.raises(DownloadError):
        distances.road_distance_matrix(np.array([[50.0, 8.0]]), np.array([[51.0, 9.0]]))
