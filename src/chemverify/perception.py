"""Perception probe: ask the model to transcribe ONE drawing into a fixed schema, then score the
transcription field by field against gold labels. This separates 'can the model read the drawing'
from 'can it reason about two drawings'.

Code scores the structured fields. The free-text fields (substituent_description,
overlaps_or_ambiguities, notes) carry ring-position detail and are judged separately, not here.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

from .stats import ratio_bootstrap

DRAWING_STYLES = ("skeletal", "condensed", "mixed", "other")
AROMATIC_DEPICTIONS = ("none", "circle", "alternating_double_bonds", "both")
INK_COLOURS = ("black", "blue", "red", "green", "mixed", "other")
LEGIBILITY = ("clear", "minor_issues", "partially_illegible", "illegible")

LIST_STR_FIELDS = ("atom_labels", "abbreviations", "r_labels", "stereo_labels", "charges")
RING_FIELD = "ring_sizes"
INT_FIELDS = ("double_bonds", "triple_bonds", "wedge_bonds", "hashed_bonds")
ENUM_FIELDS = {
    "drawing_style": DRAWING_STYLES,
    "aromatic_ring_depiction": AROMATIC_DEPICTIONS,
    "ink_colour": INK_COLOURS,
    "legibility": LEGIBILITY,
}
TEXT_FIELDS = ("substituent_description", "overlaps_or_ambiguities", "notes")
# The elements that decide whether a reproduction is faithful (ring positions are judged in text).
CRITICAL_FIELDS = ("atom_labels", "abbreviations", "r_labels", "ring_sizes", "wedge_bonds", "hashed_bonds", "stereo_labels")


def json_schema() -> dict:
    props: dict = {}
    for f in LIST_STR_FIELDS + ("overlaps_or_ambiguities",):
        props[f] = {"type": "array", "items": {"type": "string"}}
    props[RING_FIELD] = {"type": "array", "items": {"type": "integer", "minimum": 3}}
    for f in INT_FIELDS:
        props[f] = {"type": "integer", "minimum": 0}
    for f, values in ENUM_FIELDS.items():
        props[f] = {"type": "string", "enum": list(values)}
    props["substituent_description"] = {"type": "string"}
    props["notes"] = {"type": "string"}
    return {
        "type": "object",
        "properties": props,
        "required": [k for k in props if k != "notes"],
        "additionalProperties": False,
    }


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def validate(obj) -> list[str]:
    """Structural problems with a transcription (empty list = well formed)."""
    if not isinstance(obj, dict):
        return ["not a JSON object"]
    problems = [f"missing field: {k}" for k in json_schema()["required"] if k not in obj]
    for k in LIST_STR_FIELDS + ("overlaps_or_ambiguities",):
        if k in obj and not (isinstance(obj[k], list) and all(isinstance(x, str) for x in obj[k])):
            problems.append(f"{k} must be a list of strings")
    if RING_FIELD in obj and not (isinstance(obj[RING_FIELD], list) and all(_is_int(x) for x in obj[RING_FIELD])):
        problems.append("ring_sizes must be a list of integers")
    for k in INT_FIELDS:
        if k in obj and not (_is_int(obj[k]) and obj[k] >= 0):
            problems.append(f"{k} must be a non-negative integer")
    for k, allowed in ENUM_FIELDS.items():
        if k in obj and obj[k] not in allowed:
            problems.append(f"{k} must be one of {list(allowed)}")
    if "substituent_description" in obj and not isinstance(obj["substituent_description"], str):
        problems.append("substituent_description must be a string")
    return problems


_DASHES = {ord("−"): "-", ord("–"): "-", ord("—"): "-"}


def normalise_label(value) -> str:
    """Whitespace-free, Unicode-folded label (CH₃ == CH3). Case is kept: Ph and PH differ."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value))).translate(_DASHES)


def multiset_f1(pred: list, gold: list) -> float:
    p, g = Counter(pred), Counter(gold)
    if not p and not g:
        return 1.0
    common = sum((p & g).values())
    if not common:
        return 0.0
    precision, recall = common / sum(p.values()), common / sum(g.values())
    return 2 * precision * recall / (precision + recall)


def _as_list(x) -> list:
    return x if isinstance(x, list) else []


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def score(pred, gold: dict) -> dict:
    """Score one transcription against its gold transcription. Missing or malformed fields score 0."""
    problems = validate(pred)
    pred = pred if isinstance(pred, dict) else {}
    fields: dict[str, float] = {}
    within_one: dict[str, float] = {}
    for f in LIST_STR_FIELDS:
        fields[f] = multiset_f1([normalise_label(x) for x in _as_list(pred.get(f))],
                                [normalise_label(x) for x in _as_list(gold.get(f))])
    fields[RING_FIELD] = multiset_f1([x for x in _as_list(pred.get(RING_FIELD)) if _is_int(x)],
                                     [x for x in _as_list(gold.get(RING_FIELD)) if _is_int(x)])
    for f in INT_FIELDS:
        a, b = pred.get(f), gold.get(f)
        fields[f] = 1.0 if _is_int(a) and a == b else 0.0
        within_one[f] = 1.0 if _is_int(a) and _is_int(b) and abs(a - b) <= 1 else 0.0
    for f in ENUM_FIELDS:
        fields[f] = 1.0 if str(pred.get(f, "")).strip().lower() == str(gold.get(f, "")).strip().lower() else 0.0
    return {
        "fields": fields,
        "within_one": within_one,
        "overall": _mean(fields.values()),
        "critical": _mean(fields[f] for f in CRITICAL_FIELDS),
        "problems": problems,
    }


def summarise(scored: list[dict], *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean scores per image role, with a bootstrap over images (runs of one image are correlated)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in scored:
        groups[s["role"]].append(s)
        groups["all"].append(s)
    out: dict = {}
    for role, items in sorted(groups.items()):
        by_image: dict[str, list[dict]] = defaultdict(list)
        for s in items:
            by_image[s["image_key"]].append(s)
        entry: dict = {"images": len(by_image), "runs": len(items)}
        for metric in ("overall", "critical"):
            clusters = [(sum(s[metric] for s in v), len(v)) for v in by_image.values()]
            lo, hi = ratio_bootstrap(clusters, n_boot, seed)
            entry[metric] = {"mean": _mean(s[metric] for s in items), "ci95": [lo, hi]}
        entry["fields"] = {f: _mean(s["fields"][f] for s in items) for f in items[0]["fields"]}
        entry["malformed_runs"] = sum(1 for s in items if s["problems"])
        out[role] = entry
    return out


def render(summary: dict, meta: dict) -> str:
    def p(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"

    lines = [f"# Perception report: {meta.get('name', '?')}", "",
             f"Prompt `{meta.get('prompt_name')}` (sha `{meta.get('prompt_sha')}`), model `{meta.get('settings', {}).get('model')}`.", "",
             "| image role | images | runs | overall | critical elements | malformed runs |", "|---|---|---|---|---|---|"]
    for role, e in summary.items():
        ov, cr = e["overall"], e["critical"]
        lines.append(f"| {role} | {e['images']} | {e['runs']} | {p(ov['mean'])} [{p(ov['ci95'][0])}, {p(ov['ci95'][1])}] | "
                     f"{p(cr['mean'])} [{p(cr['ci95'][0])}, {p(cr['ci95'][1])}] | {e['malformed_runs']} |")
    lines += ["", "## Per-field accuracy (all images)", "", "| field | score |", "|---|---|"]
    for f, v in summary.get("all", {}).get("fields", {}).items():
        lines.append(f"| {f}{' (critical)' if f in CRITICAL_FIELDS else ''} | {p(v)} |")
    lines += ["", "Critical elements are labels, abbreviations, R-labels, ring sizes and stereo marks. "
              "Ring positions and overlaps are free text and are judged separately."]
    return "\n".join(lines) + "\n"
