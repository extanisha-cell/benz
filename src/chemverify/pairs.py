"""Build labelled image pairs from the manifest.

expected = 1 means a correct drawing (should be accepted); expected = 0 means it should be rejected.
Categories: annotated_wrong, annotated_correct, cross_molecule (reference vs a *different*
molecule's correct drawing), identity (reference vs itself; diagnostic, not gated, because a
coloured reference can legitimately fail the black-ink rule).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .paths import ROLES

CATEGORIES = ("annotated", "cross", "identity")


@dataclass(frozen=True)
class Pair:
    id: str
    molecule: str  # molecule the reference (image A) belongs to
    b_molecule: str  # molecule image B was drawn for (differs for cross_molecule)
    a: str  # image paths, relative to data/
    b: str
    a_sha: str
    b_sha: str
    expected: int
    category: str
    gated: bool

    def meta(self) -> dict:
        return {
            "pair_id": self.id,
            "molecule": self.molecule,
            "b_molecule": self.b_molecule,
            "expected": self.expected,
            "category": self.category,
            "gated": self.gated,
            "a_sha": self.a_sha[:12],
            "b_sha": self.b_sha[:12],
        }


def complete_images(manifest: dict) -> dict[str, dict[str, dict]]:
    """molecule id -> role -> image entry, only for molecules with all three images."""
    return {
        m["id"]: m["images"]
        for m in manifest["molecules"]
        if all(role in m["images"] for role in ROLES)
    }


def molecules_for(which: str, images: dict, split: dict | None) -> list[str]:
    if which == "all":
        return sorted(images)
    if which not in ("dev", "heldout"):
        raise ValueError(f"unknown split {which!r}")
    if split is None:
        raise ValueError("no split exists yet: run `split` first, or use --split all")
    return [m for m in split[which] if m in images]


def touches_heldout(molecules: list[str], split: dict | None) -> bool:
    return bool(split) and any(m in set(split["heldout"]) for m in molecules)


def build_pairs(
    images: dict,
    molecules: list[str],
    *,
    include: tuple[str, ...] = ("annotated", "cross"),
    identity_n: int = 5,
    seed: int = 7,
) -> list[Pair]:
    unknown = set(include) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"unknown pair kinds: {sorted(unknown)}")
    mols = [m for m in molecules if m in images]

    def img(m: str, role: str) -> dict:
        return images[m][role]

    pairs: list[Pair] = []
    if "annotated" in include:
        for m in mols:
            pairs.append(
                Pair(f"{m}__wrong", m, m, img(m, "ref")["file"], img(m, "wrong")["file"],
                     img(m, "ref")["sha256"], img(m, "wrong")["sha256"], 0, "annotated_wrong", True)
            )
            pairs.append(
                Pair(f"{m}__correct", m, m, img(m, "ref")["file"], img(m, "correct")["file"],
                     img(m, "ref")["sha256"], img(m, "correct")["sha256"], 1, "annotated_correct", True)
            )
    if "cross" in include and len(mols) >= 2:
        order = sorted(mols)
        random.Random(seed).shuffle(order)
        for i, m in enumerate(order):
            other = order[(i + 1) % len(order)]
            pairs.append(
                Pair(f"{m}__x__{other}", m, other, img(m, "ref")["file"], img(other, "correct")["file"],
                     img(m, "ref")["sha256"], img(other, "correct")["sha256"], 0, "cross_molecule", True)
            )
    if "identity" in include and mols:
        pick = sorted(mols)
        random.Random(seed + 1).shuffle(pick)
        for m in pick[:identity_n]:
            ref = img(m, "ref")
            pairs.append(Pair(f"{m}__self", m, m, ref["file"], ref["file"], ref["sha256"], ref["sha256"], 1, "identity", False))
    return pairs
