"""Dev / held-out split. Once frozen it is not overwritten, so held-out data stays untouched."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def make_split(
    molecules: list[str], *, n_heldout: int = 5, seed: int = 7, primary: dict[str, str] | None = None
) -> dict:
    """Seeded split. With `primary` (molecule -> main error category) the held-out set is spread
    across categories (largest categories first, one each, round-robin)."""
    ids = sorted(molecules)
    if not 0 < n_heldout < len(ids):
        raise ValueError(f"n_heldout must be between 1 and {len(ids) - 1}")
    rng = random.Random(seed)
    heldout: list[str] = []
    if primary:
        groups: dict[str, list[str]] = defaultdict(list)
        for m in ids:
            groups[primary.get(m, "unknown")].append(m)
        for members in groups.values():
            rng.shuffle(members)
        ordered = [g for _, g in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
        while len(heldout) < n_heldout:
            for members in ordered:
                if members and len(heldout) < n_heldout:
                    heldout.append(members.pop())
    else:
        shuffled = ids[:]
        rng.shuffle(shuffled)
        heldout = shuffled[:n_heldout]
    held = set(heldout)
    return {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "stratified_by": "gold primary error category" if primary else None,
        "frozen": True,
        "dev": [m for m in ids if m not in held],
        "heldout": sorted(heldout),
    }


def save_split(split: dict, path: Path, *, force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists and is frozen; pass --force to replace it (this invalidates earlier results)")
    path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")


def load_split(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def drop_heldout(records, split: dict | None) -> tuple[list[dict], int]:
    """Remove every record whose reference or drawing B belongs to a held-out molecule.

    Reports built from the result cannot reveal how the held-out molecules did, not even through
    totals. Returns (kept records, number of distinct pairs removed).
    """
    records = list(records)
    if not split:
        return records, 0
    held = set(split["heldout"])
    kept, removed = [], set()
    for rec in records:
        if rec.get("molecule") in held or rec.get("b_molecule") in held:
            removed.add(rec.get("task_id"))
        else:
            kept.append(rec)
    return kept, len(removed)


def split_of(molecule: str, split: dict | None) -> str:
    if not split:
        return "unassigned"
    if molecule in split["heldout"]:
        return "heldout"
    if molecule in split["dev"]:
        return "dev"
    return "unassigned"
