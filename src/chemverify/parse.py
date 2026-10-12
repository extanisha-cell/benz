"""Parsers for model output. The shipped format is strict: line 1 is exactly 0, 1 or 2."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class Parsed:
    decision: int | None = None  # 1 accept, 0 reject, 2 cannot verify; None when malformed
    reason: str = ""
    valid: bool = False
    problems: list[str] = field(default_factory=list)
    lenient_decision: int | None = None  # diagnostic only: what a forgiving reader would take
    data: dict | None = None  # parsed JSON for the JSON formats


_LENIENT_FIRST = re.compile(r"^\W*([012])\b")
_LENIENT_LABEL = re.compile(r"(?i)\b(?:verdict|decision|answer|score)\b\W{0,4}([012])\b")


def parse_number(text: str) -> Parsed:
    """Shipped format: line 1 is a single digit (0, 1 or 2); line 2 is a short reason."""
    out = Parsed()
    stripped = (text or "").strip()
    if not stripped:
        out.problems.append("empty output")
        return out
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    first = lines[0]
    lenient = _LENIENT_FIRST.match(first) or _LENIENT_LABEL.search(stripped)
    out.lenient_decision = int(lenient.group(1)) if lenient else None
    if first in ("0", "1", "2"):
        out.decision = int(first)
        out.valid = True
        if len(lines) > 2:
            out.problems.append(f"expected at most 2 lines, got {len(lines)}")
    else:
        out.problems.append(f"first line is not exactly 0, 1 or 2: {first[:60]!r}")
    out.reason = " ".join(lines[1:])[:2000]
    return out


def extract_json_object(text: str) -> tuple[dict | None, str | None]:
    """Find the first JSON object in `text` (tolerates a surrounding code fence or prose)."""
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", s):
        try:
            obj, _ = decoder.raw_decode(s[match.start() :])
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj, None
    return None, "no JSON object found"


_DECISIONS = {"ACCEPT": 1, "REJECT": 0, "REVIEW": 2}
_EQUIVALENCE = {"MATCH": 1, "MISMATCH": 0, "UNCERTAIN": 2}


def parse_auditor_json(text: str) -> Parsed:
    """The original auditor prompt's JSON output, mapped onto 1 / 0 / 2."""
    out = Parsed()
    obj, err = extract_json_object(text)
    if obj is None:
        out.problems.append(err or "unparseable")
        return out
    out.data = obj
    decision = str(obj.get("recommended_decision", "")).strip().upper()
    if decision in _DECISIONS:
        out.decision = _DECISIONS[decision]
        out.valid = True
    else:
        out.problems.append(f"recommended_decision missing or unknown: {decision!r}")
    out.lenient_decision = _EQUIVALENCE.get(str(obj.get("chemical_equivalence", "")).strip().upper())
    out.reason = str(obj.get("justification", ""))[:2000]
    return out


_VERDICTS = {"ACCEPT": 1, "REJECT": 0, "UNCERTAIN": 2}
_VERDICT_LINE = re.compile(r"(?i)\bverdict\b[\"'*\s:=\-]*\b(ACCEPT|REJECT|UNCERTAIN)\b")
_EQUIV = {True: 1, False: 0, None: 2}


def tri_state(value):
    """true / false / null (as a JSON value or a word) -> True / False / None; anything else -> 'bad'."""
    if value is None or isinstance(value, bool):
        return value
    return {"true": True, "false": False, "null": None, "none": None}.get(str(value).strip().lower(), "bad")


def parse_verdict_json(text: str) -> Parsed:
    """QC prompt output: verdict ACCEPT / REJECT / UNCERTAIN, as a JSON object or as 'verdict: X' text, mapped to 1 / 0 / 2.

    Fail-closed: text that names two different verdicts is invalid, and so is an ACCEPT whose chemical_equivalence
    is not true. Field names are lower-cased in `data`; `reason` is specific_evidence when the output is JSON.
    """
    out = Parsed()
    obj, _ = extract_json_object(text)
    words: set[str] = set()
    if obj is not None:
        out.data = {str(k).strip().lower(): v for k, v in obj.items()}
        if "verdict" in out.data:
            words = {str(out.data["verdict"]).strip().upper()}
    if not words:
        words = {w.upper() for w in _VERDICT_LINE.findall(text or "")}
    if len(words) != 1 or next(iter(words)) not in _VERDICTS:
        out.problems.append(f"no single verdict of ACCEPT / REJECT / UNCERTAIN found (saw {sorted(words) or 'none'})")
        return out
    verdict = next(iter(words))
    if out.data is not None:
        equiv = tri_state(out.data.get("chemical_equivalence"))
        if equiv != "bad":
            out.lenient_decision = _EQUIV[equiv]
        if verdict == "ACCEPT" and equiv is not True:
            out.problems.append("verdict ACCEPT contradicts chemical_equivalence (not true)")
            return out
        evidence = out.data.get("specific_evidence", "")
        out.reason = (evidence if isinstance(evidence, str) else json.dumps(evidence))[:2000]
    else:
        out.reason = " ".join((text or "").split())[:2000]
    out.decision = _VERDICTS[verdict]
    out.valid = True
    return out


_CLAIM = {"confirmed": 1, "refuted": 0, "inconclusive": 2}


def observation_key(text) -> str:
    """Case, spacing and punctuation folded, so two worded-differently-but-identical observations compare equal."""
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def claim_problems(data: dict) -> list[str]:
    """Why a 'confirmed' claim check cannot be believed (empty list = it can). Shared by the parser and the decision layer."""
    problems = []
    a, b = observation_key(data.get("image_a_observation")), observation_key(data.get("image_b_observation"))
    if not a or not b:
        problems.append("confirmed without an observation for both images")
    elif a == b:
        problems.append("confirmed but the two observations are identical")
    if tri_state(data.get("difference_is_chemical")) is not True:
        problems.append("confirmed but difference_is_chemical is not true")
    return problems


def parse_claim_json(text: str) -> Parsed:
    """Confirming-call output: claim_status confirmed / refuted / inconclusive, mapped to 1 / 0 / 2.

    Fail-closed: 'confirmed' needs two non-empty, different observations and difference_is_chemical true; otherwise the
    output is invalid and `decision` stays None, so nothing downstream can read it as a confirmation.
    """
    out = Parsed()
    obj, _ = extract_json_object(text)
    if obj is None:
        out.problems.append("no JSON object found")
        return out
    out.data = {str(k).strip().lower(): v for k, v in obj.items()}
    status = str(out.data.get("claim_status", "")).strip().lower()
    if status not in _CLAIM:
        out.problems.append(f"claim_status missing or unknown: {status!r}")
        return out
    evidence = out.data.get("specific_evidence", "")
    out.reason = (evidence if isinstance(evidence, str) else json.dumps(evidence))[:2000]
    if status == "confirmed":
        problems = claim_problems(out.data)
        if problems:
            out.problems.extend(problems)
            return out
    out.decision = _CLAIM[status]
    out.valid = True
    return out


def parse_perception_json(text: str) -> Parsed:
    """Perception probe output: a JSON object following perception.json_schema()."""
    out = Parsed()
    obj, err = extract_json_object(text)
    if obj is None:
        out.problems.append(err or "unparseable")
        return out
    out.data = obj
    out.valid = True
    return out


PARSERS = {
    "number": parse_number,
    "auditor_json": parse_auditor_json,
    "verdict_json": parse_verdict_json,
    "claim_json": parse_claim_json,
    "perception_json": parse_perception_json,
}
