"""Read the samples CSV and keep signed URLs out of everything that gets persisted."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, urlunsplit

from .paths import ROLES

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
ALLOWED_HOST_SUFFIXES = ("drive.google.com", ".amazonaws.com", ".googleusercontent.com")
_COLUMNS = {"ids": "id", "samples": "ref", "wrongly annotated": "wrong", "correct version": "correct"}
_DRIVE_PATH = re.compile(r"^/file/d/([A-Za-z0-9_-]+)")


@dataclass(frozen=True)
class Row:
    id: str
    ref: str
    wrong: str
    correct: str

    def url(self, role: str) -> str:
        return getattr(self, role)


def read_rows(path: str | Path) -> list[Row]:
    """Parse the CSV (columns: IDs, Samples, Wrongly annotated, Correct version)."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row")
        mapping = {
            name: _COLUMNS[name.strip().lower()]
            for name in reader.fieldnames
            if name and name.strip().lower() in _COLUMNS
        }
        missing = sorted(set(_COLUMNS.values()) - set(mapping.values()))
        if missing:
            raise ValueError(f"CSV is missing columns for {missing}; expected headers {list(_COLUMNS)}")
        rows: list[Row] = []
        seen: set[str] = set()
        for line_no, raw in enumerate(reader, start=2):
            vals = {mapping[k]: (v or "").strip() for k, v in raw.items() if k in mapping}
            if not any(vals.values()):
                continue
            if not ID_RE.match(vals["id"]):
                raise ValueError(f"line {line_no}: unsafe or empty id {vals['id']!r}")
            if vals["id"] in seen:
                raise ValueError(f"line {line_no}: duplicate id {vals['id']!r}")
            seen.add(vals["id"])
            for role in ROLES:
                if not vals[role].startswith("https://"):
                    raise ValueError(f"line {line_no}: the {role} URL for {vals['id']} must start with https://")
            rows.append(Row(**vals))
    return rows


def drive_file_id(url: str) -> str | None:
    parts = urlsplit(url)
    if (parts.hostname or "").lower() != "drive.google.com":
        return None
    match = _DRIVE_PATH.match(parts.path)
    if match:
        return match.group(1)
    ids = parse_qs(parts.query).get("id")
    return ids[0] if ids else None


def download_url(url: str) -> str:
    """The URL to GET: Drive 'view' links become direct-download links; others are unchanged."""
    file_id = drive_file_id(url)
    if file_id:
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url


def sanitize_url(url: str) -> str:
    """Drop query strings and fragments (S3 presigned signatures live there)."""
    file_id = drive_file_id(url)
    if file_id:
        return f"https://drive.google.com/file/d/{file_id}/view"
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def check_host(url: str, allowed: tuple[str, ...] = ALLOWED_HOST_SUFFIXES) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise ValueError(f"only https URLs are allowed: {sanitize_url(url)}")
    host = (parts.hostname or "").lower()
    for suffix in allowed:
        bare = suffix.lstrip(".")
        if host == bare or host.endswith("." + bare):
            return
    raise ValueError(f"host not on the allow-list: {host!r} (use --allow-host to add one)")
