"""Distance matrices: great-circle (offline) and road distances (OSRM / OpenStreetMap)."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from typing import Any

import numpy as np

from linear_opt.core.ir import FloatArray
from linear_opt.data.download import USER_AGENT, DownloadError, cache_dir

EARTH_RADIUS_KM = 6371.0088  # IUGG mean radius
OSRM_URL = "https://router.project-osrm.org"


def haversine_matrix(a: FloatArray, b: FloatArray) -> FloatArray:
    """Great-circle distances in km between rows of ``a`` and ``b`` (``[lat, lon]`` degrees)."""
    lat1, lon1 = np.radians(a[:, 0])[:, None], np.radians(a[:, 1])[:, None]
    lat2, lon2 = np.radians(b[:, 0])[None, :], np.radians(b[:, 1])[None, :]
    h = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return np.asarray(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))))


def _http_get_json(url: str, timeout: float) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def road_distance_matrix(
    a: FloatArray, b: FloatArray, *, base_url: str = OSRM_URL, timeout: float = 30.0
) -> FloatArray:
    """Driving distances in km from the OSRM ``table`` service (OpenStreetMap data, ODbL).

    One request covers the full matrix; results are cached on disk keyed by the
    coordinates, so the public demo server is hit at most once per instance.
    Please respect its usage policy, or point ``base_url`` at your own OSRM.

    Raises:
        DownloadError: On network failure or if some pair is unreachable by road.
    """
    coords = np.vstack([a, b])
    key = hashlib.sha256(np.round(coords, 5).tobytes() + base_url.encode()).hexdigest()[:16]
    cache = cache_dir() / f"osrm_{key}_{len(a)}x{len(b)}.npy"
    if cache.exists():
        return np.asarray(np.load(cache), dtype=np.float64)

    points = ";".join(f"{lon:.5f},{lat:.5f}" for lat, lon in coords)
    sources = ";".join(map(str, range(len(a))))
    destinations = ";".join(map(str, range(len(a), len(coords))))
    url = (
        f"{base_url}/table/v1/driving/{points}"
        f"?sources={sources}&destinations={destinations}&annotations=distance"
    )
    try:
        payload = _http_get_json(url, timeout)
    except OSError as exc:
        raise DownloadError(f"OSRM request failed: {exc}") from exc
    if payload.get("code") != "Ok":
        raise DownloadError(f"OSRM error: {payload.get('code')}: {payload.get('message', '')}")
    raw = payload["distances"]
    if any(d is None for row in raw for d in row):
        raise DownloadError("OSRM: some origin-destination pairs are unreachable by road")
    km = np.asarray(raw, dtype=np.float64) / 1000.0
    np.save(cache, km)
    return km
