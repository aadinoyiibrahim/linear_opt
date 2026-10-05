from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from linear_opt.data import tsplib

EUC = """NAME : toy5
COMMENT : five points
TYPE : TSP
DIMENSION : 5
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 0 0
3 3 4
2 0 1.5
4 6 8
5 10 0
EOF
"""

CVRP = """NAME : A-n4-k2
COMMENT : (toy, No of trucks: 2, Optimal value: 32)
TYPE : CVRP
DIMENSION : 4
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 10
NODE_COORD_SECTION
 1 0 0
 2 0 4
 3 3 0
 4 3 4
DEMAND_SECTION
1 0
2 6
3 6
4 3
DEPOT_SECTION
 1
 -1
EOF
"""


def test_parse_euclidean_rounds_to_nearest_integer() -> None:
    p = tsplib.parse_tsplib(EUC)
    assert (p.name, p.kind, p.dimension, p.edge_weight_type) == ("toy5", "TSP", 5, "EUC_2D")
    assert p.coords is not None
    np.testing.assert_array_equal(p.coords[1], [0, 1.5])  # node ids were unsorted
    assert p.distances[0, 2] == 5  # 3-4-5 triangle
    assert p.distances[0, 1] == 2  # nint(1.5) = 2
    assert p.distances[0, 3] == 10
    assert np.allclose(p.distances, p.distances.T)


def test_att_and_ceil_rules() -> None:
    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    # ATT: r = sqrt(100 / 10) = 3.162..., nint = 3 < r -> 4
    assert tsplib.euclidean_distances(xy, "ATT")[0, 1] == 4
    assert tsplib.euclidean_distances(np.array([[0.0, 0.0], [1.0, 1.0]]), "CEIL_2D")[0, 1] == 2
    with pytest.raises(ValueError, match="unsupported"):
        tsplib.euclidean_distances(xy, "MAN_2D")


def test_geo_distance_follows_tsplib_rule() -> None:
    # Coordinates are DDD.MM; 1 degree on the TSPLIB sphere ~ 111.3 km, +1 then truncate.
    d = tsplib.geo_distances(np.array([[10.0, 20.0], [11.0, 20.0]]))
    assert d[0, 1] == 112
    assert d[0, 0] == 0
    half = tsplib.geo_distances(np.array([[10.0, 20.0], [10.30, 20.0]]))  # 30 minutes
    assert half[0, 1] == 56


@pytest.mark.parametrize(
    ("fmt", "values"),
    [
        ("FULL_MATRIX", [0, 1, 2, 1, 0, 3, 2, 3, 0]),
        ("UPPER_ROW", [1, 2, 3]),
        ("LOWER_ROW", [1, 2, 3]),
        ("UPPER_DIAG_ROW", [0, 1, 2, 0, 3, 0]),
        ("LOWER_DIAG_ROW", [0, 1, 0, 2, 3, 0]),
        ("UPPER_COL", [1, 2, 3]),
    ],
)
def test_explicit_formats(fmt: str, values: list[float]) -> None:
    d = tsplib.explicit_distances([float(v) for v in values], 3, fmt)
    assert np.allclose(d, d.T)
    assert sorted({d[0, 1], d[0, 2], d[1, 2]}) == [1.0, 2.0, 3.0]


def test_explicit_errors() -> None:
    with pytest.raises(ValueError, match="needs"):
        tsplib.explicit_distances([1.0], 3, "UPPER_ROW")
    with pytest.raises(ValueError, match="unsupported"):
        tsplib.explicit_distances([1.0, 2.0, 3.0], 3, "WEIRD")


def test_parse_cvrp() -> None:
    p = tsplib.parse_tsplib(CVRP)
    assert p.kind == "CVRP" and p.capacity == 10 and p.depot == 0 and p.vehicles == 2
    assert p.demand is not None
    assert p.demand.tolist() == [0, 6, 6, 3]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (EUC.replace("TYPE : TSP", "TYPE : ATSP"), "unsupported TYPE"),
        (EUC.replace("EUC_2D", "XRAY1"), "unsupported EDGE_WEIGHT_TYPE"),
        (EUC.replace("DIMENSION : 5", "DIMENSION : 6"), "needs 18 numbers"),
    ],
)
def test_parse_errors(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        tsplib.parse_tsplib(text)


def test_parse_tour_and_length() -> None:
    tour = tsplib.parse_tour("NAME : t\nTYPE : TOUR\nTOUR_SECTION\n1\n3\n2\n-1\n")
    assert tour == [0, 2, 1]
    d = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    assert tsplib.tour_length(d, tour) == 6
    with pytest.raises(ValueError, match="TOUR_SECTION"):
        tsplib.parse_tour("NAME : t\n")


def test_parse_cvrp_solution() -> None:
    routes, cost = tsplib.parse_cvrp_solution("Route #1: 1 3\nRoute #2: 2\nCost 32\n")
    assert routes == [[1, 3], [2]] and cost == 32.0


def test_snapshot_uses_first_mirror_and_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from linear_opt.data.download import DownloadError

    src = tmp_path / "src"
    src.mkdir()
    (src / "toy5.tsp.gz").write_bytes(gzip.compress(EUC.encode()))  # only on mirror 2
    (src / "toy5.opt.tour").write_text("TOUR_SECTION\n1\n2\n3\n4\n5\n-1\n")  # mirror 1
    seen: list[str] = []

    def fake_fetch(url: str, *args: object, **kwargs: object) -> Path:
        seen.append(url)
        path = src / url.rsplit("/", 1)[-1]
        if not path.exists():
            raise DownloadError(f"404 {url}")
        return path

    monkeypatch.setattr(tsplib, "fetch", fake_fetch)
    fixtures = tmp_path / "fx"
    monkeypatch.setattr(tsplib, "fixture_dir", lambda lib: fixtures)
    written = tsplib.snapshot_tsplib("toy5")
    assert sorted(p.name for p in written) == ["toy5.opt.tour", "toy5.tsp"]
    assert seen[0].startswith(tsplib.TSPLIB_MIRRORS[0][0])  # GitHub mirror first
    assert any(u.endswith("toy5.tsp.gz") for u in seen)  # fell back to Heidelberg
    problem, optimum, opt_tour = tsplib.load_tsplib("toy5")
    assert opt_tour == [0, 1, 2, 3, 4]
    assert optimum == tsplib.tour_length(problem.distances, opt_tour)  # no table entry
    with pytest.raises(DownloadError, match="no mirror"):
        tsplib.fetch_tsplib_file("absent.tsp")


LISTING = """<table><tr><td><a href="https://galgos.inf.puc-rio.br/cvrplib/en/download/instance/4">A-n32-k5</a></td>
<td><a href="https://galgos.inf.puc-rio.br/cvrplib/en/download/bks/4">$784.00$</a></td></tr>
<tr><td><a href="/cvrplib/en/download/instance/5"><b>A-n33-k5</b></a></td></tr></table>"""


def test_cvrplib_links_are_found_by_name() -> None:
    inst, sol = tsplib.cvrplib_links("A-n32-k5", LISTING)
    assert inst.endswith("/download/instance/4") and sol.endswith("/download/bks/4")
    assert tsplib.cvrplib_links("A-n33-k5", LISTING)[0] == "/cvrplib/en/download/instance/5"
    with pytest.raises(ValueError, match="not found"):
        tsplib.cvrplib_links("A-n3-k5", LISTING)  # no partial matches


def test_snapshot_cvrplib_downloads_by_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    files = {
        # root-relative href, as on the real site
        "cvrplib_instances.html": '<a href="/cvrplib/en/download/instance/4">A-n4-k2</a>',
        "A-n4-k2.vrp": CVRP,
        "A-n4-k2.sol": "Route #1: 1 3\nRoute #2: 2\nCost 32\n",
    }
    urls: list[str] = []

    def fake_fetch(url: str, filename: str, **kwargs: object) -> Path:
        urls.append(url)
        path = tmp_path / filename
        path.write_text(files[filename])
        return path

    monkeypatch.setattr(tsplib, "fetch", fake_fetch)
    monkeypatch.setattr(tsplib, "fixture_dir", lambda lib: tmp_path / "fx")
    tsplib.snapshot_cvrplib("A-n4-k2")
    assert urls[1] == "https://galgos.inf.puc-rio.br/cvrplib/en/download/instance/4"
    assert urls[2] == "https://galgos.inf.puc-rio.br/cvrplib/en/download/bks/4"
    assert tsplib.load_cvrplib("A-n4-k2")[1] == 32.0


def test_snapshot_cvrplib_from_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    downloads = tmp_path / "dl"
    downloads.mkdir()
    (downloads / "A-n4-k2.vrp").write_text(CVRP)
    (downloads / "A-n4-k2.sol").write_text("Route #1: 1 3\nRoute #2: 2\nCost 32\n")
    fixtures = tmp_path / "fx"
    monkeypatch.setattr(tsplib, "fixture_dir", lambda lib: fixtures)
    tsplib.snapshot_cvrplib("A-n4-k2", from_dir=downloads)
    _, cost, routes = tsplib.load_cvrplib("A-n4-k2")
    assert cost == 32.0 and routes == [[1, 3], [2]]
    (fixtures / "A-n4-k2.sol").unlink()
    assert tsplib.load_cvrplib("A-n4-k2")[1] == 32.0  # falls back to the COMMENT
    with pytest.raises(FileNotFoundError):
        tsplib.snapshot_cvrplib("missing", from_dir=downloads)
