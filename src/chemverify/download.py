"""Download the sample images into one folder per molecule and record what was fetched.

Downloaded files are untrusted data: only image bytes are kept, the format is verified from
magic bytes (not the URL or Content-Type), and nothing is ever executed or imported.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .imageinfo import ImageInfo, sniff
from .manifest import ALLOWED_HOST_SUFFIXES, Row, check_host, download_url, sanitize_url
from .paths import ROLES

MAX_BYTES = 25_000_000
Fetcher = Callable[[str], "tuple[bytes, str]"]


class DownloadError(Exception):
    pass


def urllib_fetch(url: str, *, timeout: float = 60, retries: int = 3, sleep=time.sleep) -> tuple[bytes, str]:
    """GET a URL; returns (bytes, content_type). Error messages never contain the query string."""
    shown = sanitize_url(url)
    for attempt in range(retries):
        last = attempt == retries - 1
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "chemverify/0.1"})
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                data = resp.read(MAX_BYTES + 1)
                content_type = resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and not last:
                sleep(2**attempt)
                continue
            raise DownloadError(f"HTTP {exc.code} for {shown}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if not last:
                sleep(2**attempt)
                continue
            reason = getattr(exc, "reason", exc)
            raise DownloadError(f"network error for {shown}: {type(reason).__name__}") from None
        if len(data) > MAX_BYTES:
            raise DownloadError(f"{shown} is larger than {MAX_BYTES} bytes")
        return data, content_type
    raise DownloadError(f"could not fetch {shown}")  # pragma: no cover


def atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _entry(path: Path, data_root: Path, data: bytes, info: ImageInfo, url: str) -> dict:
    return {
        "file": path.relative_to(data_root).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        **info.as_dict(),
        "source": sanitize_url(url),
    }


def download_all(
    rows: list[Row],
    dest_root: Path,
    *,
    fetch: Fetcher = urllib_fetch,
    workers: int = 4,
    force: bool = False,
    allowed_hosts: tuple[str, ...] = ALLOWED_HOST_SUFFIXES,
    log: Callable[[str], None] = print,
) -> tuple[dict, list[str]]:
    """Fetch ref/wrong/correct for every row. Returns (manifest, problems)."""
    dest_root = Path(dest_root)
    data_root = dest_root.parent
    results: dict[tuple[str, str], dict] = {}
    problems: list[str] = []

    def one(row: Row, role: str) -> dict:
        url = row.url(role)
        folder = dest_root / row.id  # one folder per molecule
        folder.mkdir(parents=True, exist_ok=True)
        if not force:
            for existing in sorted(folder.glob(f"{role}.*")):
                data = existing.read_bytes()
                info = sniff(data)
                if info and info.complete:
                    return _entry(existing, data_root, data, info, url)
        check_host(url, allowed_hosts)
        data, content_type = fetch(download_url(url))
        if "text/html" in content_type.lower():
            raise DownloadError("received an HTML page instead of an image (file not shared, or needs confirmation)")
        info = sniff(data)
        if info is None:
            raise DownloadError("downloaded bytes are not a recognised image")
        for stale in folder.glob(f"{role}.*"):
            stale.unlink()
        path = folder / f"{role}.{info.extension}"
        atomic_write(path, data)
        return _entry(path, data_root, data, info, url)

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, row, role): (row.id, role) for row in rows for role in ROLES}
        for future in cf.as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
                log(f"ok    {key[0]}/{key[1]}")
            except Exception as exc:  # report and carry on with the other files
                problems.append(f"{key[0]}/{key[1]}: {exc}")
                log(f"FAIL  {key[0]}/{key[1]}: {exc}")

    molecules = []
    for row in rows:
        images = {role: results[(row.id, role)] for role in ROLES if (row.id, role) in results}
        molecules.append({"id": row.id, "images": images, "complete": len(images) == len(ROLES)})
    manifest = {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "molecules": molecules,
    }
    return manifest, problems
