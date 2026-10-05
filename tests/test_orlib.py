from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from linear_opt.data import orlib

# m=2 warehouses, n=3 customers; line breaks deliberately irregular.
MINI = """ 2 3
 10 100.5
 20 200.0
 4
 1.0 2.0
 5 3.0
 4.0 6
 7.0 8.0
"""


def test_parse_cap_shapes_and_orientation() -> None:
    data = orlib.parse_cap(MINI, "mini")
    assert data.shape == (2, 3)
    np.testing.assert_array_equal(data.capacity, [10, 20])
    np.testing.assert_array_equal(data.fixed_cost, [100.5, 200.0])
    np.testing.assert_array_equal(data.demand, [4, 5, 6])
    # cost[i, j]: rows are warehouses, columns customers
    np.testing.assert_array_equal(data.cost, [[1, 3, 7], [2, 4, 8]])


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("2 3 1 2", "expected"),
        ("", "empty"),
        ("2 1 capacity 7500 5000 7500 3 1 2", "non-numeric"),
    ],
)
def test_parse_cap_rejects_malformed(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        orlib.parse_cap(text)


def test_parse_capopt() -> None:
    text = "Optimal solution values\n cap41   1040444.375\ncap131 793439.562\n junk line\n"
    assert orlib.parse_capopt(text) == {
        "cap41": 1040444.375,
        "cap131": 793439.562,
    }


def test_instance_catalogue() -> None:
    assert len(orlib.CAP_IDS) == 37
    assert orlib.CAP_IDS[:5] == ("cap41", "cap42", "cap43", "cap44", "cap51")
    with pytest.raises(ValueError, match="unknown instance"):
        orlib.load_cap("cap99")


def test_load_prefers_fixture_then_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixtures, cache = tmp_path / "fx", tmp_path / "cache"
    fixtures.mkdir()
    cache.mkdir()
    (cache / "cap42.txt").write_text(MINI)
    (fixtures / "cap41.txt").write_text(MINI)
    (fixtures / "capopt.txt").write_text("cap41 123.5\n")
    fetched: list[str] = []

    def fake_fetch(url: str) -> Path:
        fetched.append(url)
        return cache / url.rsplit("/", 1)[-1]

    monkeypatch.setattr(orlib, "fixture_dir", lambda: fixtures)
    monkeypatch.setattr(orlib, "fetch", fake_fetch)
    assert orlib.load_cap("cap41").shape == (2, 3) and fetched == []
    assert orlib.known_optimum("cap41") == 123.5
    assert orlib.load_cap("cap42").shape == (2, 3)
    assert fetched == [orlib.BASE_URL + "cap42.txt"]


def test_snapshot_validates_and_copies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "cap41.txt").write_text(MINI)
    (src / "capopt.txt").write_text("cap41 1.0\n")
    monkeypatch.setattr(orlib, "fetch", lambda url: src / url.rsplit("/", 1)[-1])
    written = orlib.snapshot(["cap41"], tmp_path / "out")
    assert sorted(p.name for p in written) == ["cap41.txt", "capopt.txt"]
    (src / "cap41.txt").write_text("2 3 1")
    with pytest.raises(ValueError, match="expected"):
        orlib.snapshot(["cap41"], tmp_path / "out2")
