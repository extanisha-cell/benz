"""Project locations. Scripts live in src/; downloaded (untrusted) data lives only under data/."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
IMAGES = DATA / "images"
GOLD = DATA / "gold"
PROMPTS = ROOT / "prompts"
RUNS = ROOT / "runs"
MANIFEST = DATA / "manifest.json"
SPLITS = DATA / "splits.json"
HELDOUT_LOG = RUNS / "_heldout_access.log"

ROLES = ("ref", "wrong", "correct")
