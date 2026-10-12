"""Prompt files are sent verbatim (surrounding whitespace stripped); the hash identifies a version."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptSpec:
    name: str
    path: str
    text: str
    sha: str


def load_prompt(path: str | Path) -> PromptSpec:
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig").strip()  # utf-8-sig: a Windows BOM must never reach the model
    if not text:
        raise ValueError(f"prompt file is empty: {p}")
    return PromptSpec(p.stem, str(p), text, hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])
