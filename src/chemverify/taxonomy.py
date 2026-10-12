"""Shared vocabulary for error categories (gold-label agents, reports, perception scoring)."""
from __future__ import annotations

ERROR_CATEGORIES = {
    "missing_atom_or_group": "An atom, group, substituent or ring present in the reference is absent.",
    "extra_atom_or_group": "Something is drawn that is not in the reference.",
    "wrong_atom_or_label": "An atom symbol or label differs (for example N instead of O).",
    "bond_order": "A single, double or triple bond is drawn with the wrong order.",
    "connectivity": "Atoms are joined differently, or a bond is missing or added.",
    "ring_size_or_fusion": "A ring has a different size, or ring fusion differs.",
    "substituent_position": "A group is attached at a different ring or chain position.",
    "stereochemistry": "Wedge/hash bonds, E/Z, R/S or cis/trans differ or are missing.",
    "abbreviation_changed": "An abbreviation was expanded, compressed or altered (Ph written out, PH changed, ...).",
    "label_changed": "A label such as R1 is missing, renamed or misplaced.",
    "charge_or_hydrogen_count": "Charge, radical or explicit hydrogen count differs.",
    "legibility": "A critical symbol or stroke is unreadable or ambiguous.",
    "ink_colour": "Drawn in a colour other than black.",
    "overlap_handling": "Overlapping elements reproduced in a way that hurts legibility.",
    "other": "None of the above.",
}
