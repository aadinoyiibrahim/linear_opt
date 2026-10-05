"""Cached, verified downloads of external datasets."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from linear_opt import __version__

USER_AGENT = f"linear-opt/{__version__} (+https://github.com/aadinoyiibrahim/linear_opt)"


class DownloadError(RuntimeError):
    """A dataset could not be downloaded or failed verification."""


def cache_dir() -> Path:
    """Directory for downloaded data: ``$LINEAR_OPT_DATA_DIR`` or ``~/.cache/linear_opt``."""
    root = os.environ.get("LINEAR_OPT_DATA_DIR")
    path = Path(root) if root else Path.home() / ".cache" / "linear_opt"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256sum(path: Path) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(
    url: str,
    filename: str | None = None,
    *,
    sha256: str | None = None,
    force: bool = False,
    timeout: float = 60.0,
) -> Path:
    """Download ``url`` into the cache (once) and return the local path.

    The file is written to ``<name>.part`` and renamed only after the optional
    checksum matches, so an interrupted download never leaves a corrupt file.

    Raises:
        DownloadError: On network failure or checksum mismatch.
    """
    target = cache_dir() / (filename or url.rsplit("/", 1)[-1])
    if target.exists() and not force and (sha256 is None or sha256sum(target) == sha256):
        return target
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            partial.open("wb") as out,
        ):
            while chunk := response.read(1 << 20):
                out.write(chunk)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"could not download {url}: {exc}") from exc
    if sha256 is not None and (actual := sha256sum(partial)) != sha256:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"checksum mismatch for {url}: expected {sha256}, got {actual}")
    partial.replace(target)
    return target
