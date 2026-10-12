"""One chemical-equivalence policy, shared by every model call and by the final decision.

The text sent to the model (`system_prompt`) and the rules the decision layer applies (the category tables,
`canonical_label`, `classify_text`, the quorum constants) are all defined in this file, so the prompt and the
code that judges the answers cannot drift apart. `tests/test_policy_decision.py` checks that the prompt file
in prompts/ is exactly what `system_prompt()` renders.

Two kinds of difference between a reference drawing and an annotator's drawing:

* CHEMICAL: the molecule is not the same (different element, connectivity, bond order, ring, substituent
  position, charge, isotope, stereochemistry, or an atom/group present in only one drawing). Only these can
  justify a REJECT, and only with a location and what each drawing shows.
* COSMETIC / NOTATION: the same molecule drawn differently (aromatic circle vs Kekule, an abbreviation written
  out, implicit vs explicit atoms, handwriting, rotation, scale, ...). These are recorded, never a reason to reject.

Whether the annotation guidelines (black ink, no abbreviation changes, legibility) were followed is a different
question from chemical identity. This policy does not decide it; notation differences are kept in the output so
that nothing is lost.
"""
from __future__ import annotations

import re
import unicodedata

POLICY_VERSION = "chem-equiv-1"       # version of the PROMPT text (system_prompt); changing it changes what the model sees
DECISION_VERSION = "chem-decision-2"  # version of the RULES in decision.py; deliberately not part of any prompt
SCHEMA_VERSION = "2"                  # version of the results CSV layout written by pipeline_google10.py

# Evidence thresholds used by decision.decide. Changing them changes every decision, so they live here.
QUORUM = 2                # structured, fully verified ACCEPT calls needed to accept
MAX_STRUCTURED_CALLS = 3  # structured calls per task before an unresolved case is left for a person
MAX_CONFIRMATION_CALLS = 1  # confirming calls per task; they test the leading reported difference only

# The authoritative policy is this file. prompts/chem_equiv_policy_v1.txt and prompts/chem_equiv_confirm_v1.txt are
# generated from it (`pipeline_google10.py policy --write`) and a test checks that they match. Drafts that nothing
# reads live in prompts/archive/.

CHECKLIST_ASPECTS = ("atoms", "bonds", "connectivity", "substituents", "charges", "stereochemistry")
CHECKLIST_STATES = ("match", "mismatch", "unreadable", "not_present")
OPTIONAL_ASPECTS = ("charges", "stereochemistry")  # may be "not_present" when neither drawing has any

CHEMICAL_CATEGORIES = {
    "element_mismatch": "a different element at the same position (for example N where the reference has O)",
    "connectivity": "atoms bonded to different partners, or a ring closed or opened differently",
    "bond_order": "a single, double or triple bond where the reference has another order (an aromatic circle versus Kekule is NOT a bond-order difference)",
    "ring_size_or_fusion": "a ring with a different number of members, or rings fused differently",
    "substituent_position": "a group attached at a different position (ortho/meta/para, locant numbers)",
    "missing_or_extra_atom_or_group": "an atom, group, ring or substituent present in only one drawing, after expanding abbreviations",
    "formal_charge": "a charge, radical or lone-pair marking that differs",
    "isotope": "an isotope label that differs or is present in only one drawing",
    "stereochemistry": "wedge/hash bonds, double-bond geometry or a stereocentre that differs or is present in only one drawing",
    "other_chemical": "any other difference that changes the identity of the molecule",
}
COSMETIC_CATEGORIES = {
    "aromatic_depiction": "an aromatic circle versus Kekule alternating double bonds",
    "abbreviation_expansion": "an abbreviation written out, or a group drawn out in full (Me/CH3, Et/CH2CH3, OEt/OCH2CH3, CF3/C(F)3, Ph/a drawn phenyl ring)",
    "implicit_explicit_atoms": "implicit versus explicit carbons or hydrogens (H3C for an implicit methyl, a CH2 label at a vertex)",
    "label_format": "equivalent label notation (CH3 versus CH3 with subscripts, H3C versus CH3, reversed group order)",
    "drawing_style": "handwriting, line quality, skeletal versus condensed style, rotation, scale, spacing, bond angles, paper background, ink colour",
}

_CATEGORY_ALIASES = {
    "stereo": "stereochemistry", "stereochemical": "stereochemistry", "chirality": "stereochemistry",
    "element": "element_mismatch", "atom_type": "element_mismatch", "wrong_element": "element_mismatch",
    "connectivity_mismatch": "connectivity", "bonding": "connectivity",
    "bond_order_mismatch": "bond_order", "bond_orders": "bond_order",
    "ring_size": "ring_size_or_fusion", "ring_fusion": "ring_size_or_fusion", "ring": "ring_size_or_fusion",
    "substituent": "substituent_position", "substitution_position": "substituent_position",
    "positional": "substituent_position", "regiochemistry": "substituent_position",
    "missing_atom": "missing_or_extra_atom_or_group", "extra_atom": "missing_or_extra_atom_or_group",
    "missing_group": "missing_or_extra_atom_or_group", "extra_group": "missing_or_extra_atom_or_group",
    "missing_or_extra_group": "missing_or_extra_atom_or_group", "missing_or_extra_atom": "missing_or_extra_atom_or_group",
    "charge": "formal_charge", "charges": "formal_charge",
    "aromaticity": "aromatic_depiction", "aromatic_ring_depiction": "aromatic_depiction", "kekule": "aromatic_depiction",
    "abbreviation": "abbreviation_expansion", "abbreviations": "abbreviation_expansion",
    "abbreviation_change": "abbreviation_expansion", "notation": "label_format", "style": "drawing_style",
    "handwriting": "drawing_style", "orientation": "drawing_style",
}

RULES = (
    "Compare molecular identity, not drawing style.",
    "Treat chemically equivalent abbreviations and expanded groups as identical (Me/CH3, Et/CH2CH3, CF3/C(F)3, "
    "OEt/OCH2CH3, a phenyl ring drawn out in full versus Ph).",
    "Treat aromatic circles and valid Kekule representations as equivalent.",
    "Ignore rotations, translations, scale, bond-angle changes, handwriting, line quality, spacing, paper background "
    "and ink colour when they preserve chemical identity.",
    "Verify atom connectivity, elements, formal charges, bond orders, substituent positions, isotope labels, and stereochemistry.",
    "Reject only when there is an identifiable chemical mismatch, and report it as a discrepancy with its location.",
    "Do not reject for a cosmetic or notation-only difference. Record it under notation_differences instead; it never "
    "changes the verdict.",
    "If a critical atom, bond, or stereochemical relationship is unreadable, return UNCERTAIN rather than guessing.",
    "Apply the same criteria consistently to every task.",
)

_OUTPUT_SPEC = """Return exactly one JSON object and nothing else, with these fields:
{
  "verdict": "ACCEPT" | "REJECT" | "UNCERTAIN",
  "chemical_equivalence": true | false | null,
  "checklist": {"atoms": S, "bonds": S, "connectivity": S, "substituents": S, "charges": S, "stereochemistry": S},
  "discrepancies": [{"category": "<chemical category>", "location": "<where in the drawing>", "reference": "<what Image A shows>", "annotated": "<what Image B shows>"}],
  "notation_differences": [{"category": "<notation category>", "detail": "<what differs>"}],
  "unreadable_regions": ["<region you could not read>"],
  "mismatch_type": "<chemical category of the main discrepancy, or null>",
  "specific_evidence": "<one or two sentences>",
  "confidence": <number from 0 to 1: how sure you are of the verdict>,
  "requires_review": true | false
}
Field rules:
- Each S in "checklist" is "match", "mismatch" or "unreadable"; charges and stereochemistry may also be "not_present" when neither drawing has any.
- ACCEPT requires chemical_equivalence true, every checklist entry "match" or "not_present", and an empty "discrepancies" list.
- REJECT requires chemical_equivalence false and at least one entry in "discrepancies", each with a chemical category, a location, and what each drawing shows.
- UNCERTAIN requires chemical_equivalence null and at least one unreadable region.
- "notation_differences" may be non-empty with any verdict."""


def _bullets(table: dict) -> str:
    return "\n".join(f"- {name}: {text}" for name, text in table.items())


def system_prompt() -> str:
    """The single system prompt sent with every call (first calls and escalations alike)."""
    rules = "\n".join(f"{i}. {rule}" for i, rule in enumerate(RULES, start=1))
    return (
        "You are a chemistry annotation quality-control verifier.\n\n"
        "Your task is to determine whether two molecular structure images encode the same chemical structure. "
        "Image A is the reference and Image B is the annotator's drawing.\n\n"
        f"Evaluation rules (policy {POLICY_VERSION}):\n{rules}\n\n"
        f"Chemical discrepancy categories (the only valid grounds for REJECT):\n{_bullets(CHEMICAL_CATEGORIES)}\n\n"
        f"Notation categories (never grounds for REJECT):\n{_bullets(COSMETIC_CATEGORIES)}\n\n"
        f"{_OUTPUT_SPEC}\n\n"
        "Do not invent mismatches. A REJECT verdict must identify a specific chemical discrepancy and its location.\n"
    )


# --------------------------------------------------------------------------- confirmation prompt

CONFIRM_STATUSES = ("confirmed", "refuted", "inconclusive")

_CONFIRM_OUTPUT_SPEC = """Return exactly one JSON object and nothing else, with these fields:
{
  "claim_status": "confirmed" | "refuted" | "inconclusive",
  "image_a_observation": "<what Image A shows at the stated location, described from Image A alone>",
  "image_b_observation": "<what Image B shows at the stated location, described from Image B alone>",
  "difference_is_chemical": true | false | null,
  "unreadable_regions": ["<region you could not read>"],
  "specific_evidence": "<one or two sentences>",
  "confidence": <number from 0 to 1: how sure you are of claim_status>
}
Field rules:
- Look at the stated location in Image A first and write image_a_observation from Image A alone. Then look at the same location in Image B and write image_b_observation from Image B alone. If the claim names a count (for example the number of ring members), count it in each image. If it names a position, state the position in each image.
- "confirmed" requires that your two observations differ in the feature named by the claim AND that the difference changes the identity of the molecule (difference_is_chemical true).
- "refuted" means your observations agree on that feature, or the difference is only a notation difference (difference_is_chemical false).
- "inconclusive" means the stated location is unreadable or ambiguous in either image, or you cannot tell the two apart; list it under unreadable_regions."""


def confirmation_prompt() -> str:
    """The single system prompt of the confirming call. It shares the rules and category tables of system_prompt(),
    so a claim is judged by the same policy that produced it; only the task differs (test one claim, do not judge the pair)."""
    rules = "\n".join(f"{i}. {rule}" for i, rule in enumerate(RULES, start=1))
    return (
        "You are a chemistry annotation quality-control verifier. A previous review of two molecular structure images "
        "reported a possible chemical difference. Your task is to test that one claim independently, not to take it on trust.\n\n"
        "Image A is the reference and Image B is the annotator's drawing. The claim is given in the user message together with the images.\n\n"
        f"Evaluation rules (policy {POLICY_VERSION}):\n{rules}\n\n"
        f"Chemical discrepancy categories (the only differences that can confirm a claim):\n{_bullets(CHEMICAL_CATEGORIES)}\n\n"
        f"Notation categories (a difference of this kind never confirms a claim):\n{_bullets(COSMETIC_CATEGORIES)}\n\n"
        "Treat the claim as a hypothesis. It may be right or wrong. Reviews of hand-drawn structures sometimes miscount ring members, "
        "misread one letter of a label, or misjudge where a substituent sits; they also sometimes find a real error. "
        "Do not dismiss the claim without looking, and do not agree with it merely because it is stated. "
        "A claim you cannot see in the images is refuted or inconclusive, never confirmed.\n\n"
        f"{_CONFIRM_OUTPUT_SPEC}\n"
    )


def claim_message(claim: dict) -> str:
    """The per-task user message that carries one claim. `claim` has category, location, reference, annotated."""
    return (
        "Claim to test.\n"
        f"Category: {claim.get('category', '')}\n"
        f"Location: {claim.get('location', '')}\n"
        f"The earlier review stated that Image A shows: {claim.get('reference', '') or '(not stated)'}\n"
        f"The earlier review stated that Image B shows: {claim.get('annotated', '') or '(not stated)'}\n"
        "Examine both images yourself and answer in the required JSON format."
    )


# --------------------------------------------------------------------------- category handling


def normalise_category(raw: str) -> str:
    """A model-written category -> a key of CHEMICAL_CATEGORIES or COSMETIC_CATEGORIES, or '' when unknown."""
    slug = re.sub(r"[^a-z0-9]+", "_", (raw or "").strip().lower()).strip("_")
    slug = _CATEGORY_ALIASES.get(slug, slug)
    return slug if slug in CHEMICAL_CATEGORIES or slug in COSMETIC_CATEGORIES else ""


# --------------------------------------------------------------------------- label normalisation

_DASHES = {ord("−"): "-", ord("–"): "-", ord("—"): "-"}

# canonical form -> spellings of the same group (either reading order). Short on purpose: it normalises labels
# for the cross-check column and for judging a model's own "reference vs annotated" pair. NC is deliberately absent
# (an isonitrile is not a nitrile).
_SYNONYMS = {
    "CH3": ("Me", "CH3", "H3C"),
    "C2H5": ("Et", "C2H5", "CH2CH3", "H5C2", "H3CH2C"),
    "OCH3": ("OMe", "MeO", "OCH3", "CH3O", "H3CO"),
    "OC2H5": ("OEt", "EtO", "OC2H5", "OCH2CH3", "C2H5O", "CH3CH2O", "H5C2O"),
    "NHCH3": ("NHMe", "MeNH", "NHCH3", "CH3NH", "HNMe", "H3CHN"),
    "N(CH3)2": ("NMe2", "Me2N", "N(CH3)2", "(CH3)2N", "N(Me)2"),
    "CO2CH3": ("COOMe", "CO2Me", "MeOOC", "MeO2C", "COOCH3", "CO2CH3", "H3COOC", "H3CO2C"),
    "CO2C2H5": ("COOEt", "CO2Et", "EtOOC", "EtO2C", "COOC2H5", "CO2C2H5"),
    "CO2H": ("COOH", "CO2H", "HOOC", "HO2C"),
    "CF3": ("CF3", "F3C", "C(F)3"),
    "OH": ("OH", "HO"),
    "NH2": ("NH2", "H2N"),
    "SH": ("SH", "HS"),
    "NO2": ("NO2", "O2N"),
    "C6H5": ("Ph", "C6H5"),
    "CO2C6H5": ("PhOOC", "COOPh", "CO2Ph", "PhO2C"),
}
_CANON = {spelling: canon for canon, spellings in _SYNONYMS.items() for spelling in spellings}
ABBREVIATION_TOKENS = tuple(sorted({s for spellings in _SYNONYMS.values() for s in spellings if s[0].isupper()} | {"iPr", "tBu", "Bn", "Ac", "Bu", "Pr"}))


def canonical_label(label) -> str:
    """Whitespace-free, Unicode-folded label with equivalent group spellings mapped together (Me == CH3 == H3C).
    Case is kept (Ph and PH differ); labels not in the table are returned unchanged."""
    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(label))).translate(_DASHES)
    return _CANON.get(text, text)


# --------------------------------------------------------------------------- free-text classification

_COSMETIC_PATTERNS = {
    "aromatic_depiction": re.compile(r"(?i)\bcircles?\b|kekul|alternating\s+double|delocali[sz]ed|aromatic\s+(ring\s+)?(depiction|representation|notation)"),
    "implicit_explicit_atoms": re.compile(r"(?i)\bimplicit\b|explicitly\s+(writes?|draws?|shows?|labels?)|\bH3C\b"),
    "drawing_style": re.compile(r"(?i)handwrit|\brotat|orientation|bond\s+angles?|\bspacing\b|\bstyle\b|\bink\b"),
    "label_format": re.compile(r"(?i)subscript|superscript|reversed\s+order"),
}
_ABBREVIATION_WORD = re.compile(r"(?i)abbreviat")
_EXPANSION_WORD = re.compile(r"(?i)\bexpand\w*|written\s+out|spelled\s+out|fully\s+drawn|drawn\s+out|\bcompress\w*")
_ABBREVIATION_TOKEN = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(t) for t in ABBREVIATION_TOKENS) + r")(?![a-z])")
_CHEMICAL_PATTERN = re.compile(
    r"(?i)\b(pentagon|hexagon|heptagon|(five|six|seven)[- ]?membered|ring\s+size)\b|\b(ortho|meta|para)\b|connectivity"
    r"|bonded\s+to\s+(a\s+)?different|bond\s+order|missing\s+(an?\s+)?(atom|bond|group|ring|substituent|methyl|carbon)"
    r"|extra\s+(an?\s+)?(atom|bond|group|ring|substituent|methyl|carbon)|stereo\w*|\bwedge|\bhash(ed)?\b|chiral"
    r"|\b[RS]\s+configuration|\bE/Z\b|\bcis\b|\btrans\b|\bcharge|isotope|instead\s+of\s+an?\s+(single|double|triple)\b|\bfused\b|\bfusion\b"
    r"|different\s+(atom|element)|\bpositions?\s+of\b|substituent\s+position"
)


def classify_text(text: str) -> tuple[set[str], bool]:
    """Heuristic label for a free-text reason: (notation categories it mentions, whether it also makes a structural claim).

    Used only to label the old digit-plus-reason responses and to spot structural claims in them; the structured
    path relies on the model's own categories and locations instead. A text that sounds structural is treated as
    an open allegation by the decision layer, so a wrong label here can cost an extra call but never an accept.
    """
    text = text or ""
    cosmetic = {name for name, pattern in _COSMETIC_PATTERNS.items() if pattern.search(text)}
    if _ABBREVIATION_WORD.search(text) or (_EXPANSION_WORD.search(text) and _ABBREVIATION_TOKEN.search(text)):
        cosmetic.add("abbreviation_expansion")
    return cosmetic, bool(_CHEMICAL_PATTERN.search(text))
