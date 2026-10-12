"""Deterministic final-decision layer (rules version `policy.DECISION_VERSION`).

Gemini's calls are never trusted at face value. Each policy-prompt call is turned into a `Judgement` (what it can
support under the shared policy); an optional confirming call is turned into a `Confirmation`; fixed rules combine them
into one `Decision`; `finalize` writes the explanation from that decision and runs a consistency gate. No model is
consulted here, so the same recorded responses always give the same decision.

Per-call votes (policy-prompt calls)
    ACCEPT     checklist complete and clean (verified); an ACCEPT without a complete checklist is counted but never enough
    REJECT     a chemical discrepancy that names a location
    UNCERTAIN  an unreadable region, or a call that contradicts itself
    EXCLUDED   a reject the policy does not allow to count: notation-only grounds, or no concrete evidence
    INVALID    blocked, empty, unparseable, or an ACCEPT that contradicts its own evidence

Final outcome, first rule that applies
    1  graph check says mismatch and a supported reject exists       -> REJECT    graph_confirmed_reject
    2  graph check says mismatch, no supported reject                -> UNCERTAIN graph_vlm_conflict
    3  supported reject, and a verified accept or a graph match      -> UNCERTAIN contested_reject
       (repeated rejects do not override an accept: the calls share one model and one image)
    4  supported reject, nothing contradicts it: it needs a confirming call (a different prompt that tests the leading
       claim image by image). Confirmed -> REJECT confirmed_reject; refuted -> UNCERTAIN checks_disagree;
       inconclusive or invalid -> UNCERTAIN reject_not_confirmed; not run -> UNCERTAIN confirmation_not_run;
       not run yet -> UNCERTAIN reject_pending_confirmation (transient: it never reaches an exported row)
    5  any UNCERTAIN vote                                              -> UNCERTAIN unreadable_or_uncertain
    6  verified accepts (plus a graph match) >= QUORUM                -> ACCEPT    verified_equivalence
    7  otherwise                                                       -> UNCERTAIN insufficient_evidence
UNCERTAIN is never turned into ACCEPT or REJECT by this layer. A REJECT is published only when its claim was confirmed
by a second, different check (or by a graph comparison); the confirming check is still Gemini, so it is evidence, not proof.
Model confidence is stored but never enters a rule.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

from . import policy
from .graphcheck import NOT_PROVIDED, GraphResult
from .parse import claim_problems, extract_json_object, tri_state

ACCEPT, REJECT, UNCERTAIN = "ACCEPT", "REJECT", "UNCERTAIN"
VOTES = (ACCEPT, REJECT, UNCERTAIN, "EXCLUDED", "INVALID")


@dataclass
class Judgement:
    vote: str  # one of VOTES
    verified: bool = False  # ACCEPT with a complete, clean checklist
    evidence: list = field(default_factory=list)  # supported discrepancies (vote REJECT)
    notation: list = field(default_factory=list)  # notation categories noted
    flags: list = field(default_factory=list)
    confidence: float | None = None  # the model's own confidence, 0..1, if it gave a number
    run_idx: int | None = None


@dataclass
class Confirmation:
    status: str  # confirmed | refuted | inconclusive | invalid
    claim_key: str = ""  # the claim this call tested (see claim_key)
    observations: dict = field(default_factory=dict)  # {"reference": ..., "annotated": ...}
    flags: list = field(default_factory=list)
    confidence: float | None = None
    statement: str = ""
    run_idx: int | None = None


@dataclass
class Decision:
    outcome: str
    rule: str
    failure_categories: list
    evidence: list
    notation_differences: list
    flags: list
    needs_escalation: bool  # another policy-prompt call may resolve it
    needs_confirmation: bool  # the leading claim still needs its confirming call
    requires_review: bool
    votes: dict
    accept_share: float | None  # verified accepts / policy-prompt calls
    model_confidence: float | None  # mean stated confidence of the calls that match the outcome; never used in a rule
    structured_calls: int
    verified_accepts: int
    graph: str
    claim: dict | None  # the leading reported difference (most calls citing it)
    evidence_status: str  # none | asserted | confirmed | graph_confirmed | refuted | inconclusive
    confirmation: dict | None  # the applicable confirming call, if any
    calls: list = field(default_factory=list)  # per-call audit trail
    explanation: str = ""
    consistency_status: str = "unchecked"  # ok | conflict | unchecked
    consistency: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# --------------------------------------------------------------------------- one call -> one judgement


def _confidence(value) -> float | None:
    """The model's own confidence as 0..1. Words ("high") and anything else are not turned into numbers."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        value = float(text[:-1]) / 100 if text.endswith("%") and text[:-1].replace(".", "", 1).isdigit() else (
            float(text) if text.replace(".", "", 1).isdigit() else None)
        if value is None:
            return None
    if not isinstance(value, (int, float)) or value < 0:
        return None
    return round(value / 100 if value > 1 else float(value), 4) if value <= 100 else None


def _text(d: dict, *keys: str) -> str:
    for key in keys:
        if d.get(key):
            return " ".join(str(d[key]).split())
    return ""


def _checklist(raw) -> dict:
    out: dict = {}
    if isinstance(raw, dict):
        lowered = {str(k).strip().lower(): v for k, v in raw.items()}
        for aspect in policy.CHECKLIST_ASPECTS:
            state = str(lowered.get(aspect, "")).strip().lower().replace(" ", "_").replace("-", "_")
            if state in policy.CHECKLIST_STATES:
                out[aspect] = state
    return out


def claim_key(category: str, location: str) -> str:
    """Identity of a reported difference: its category and its location with case and spacing folded."""
    return f"{category}|{' '.join((location or '').lower().split())}"


def _discrepancies(raw) -> list[dict]:
    """Normalise the model's discrepancy list and mark entries that are really notation differences."""
    out = []
    for item in raw if isinstance(raw, list) else []:
        d = {str(k).strip().lower(): v for k, v in item.items()} if isinstance(item, dict) else {"detail": str(item)}
        entry = {"category": policy.normalise_category(_text(d, "category", "type")), "location": _text(d, "location", "where"),
                 "reference": _text(d, "reference", "image_a"), "annotated": _text(d, "annotated", "image_b"),
                 "detail": _text(d, "detail", "evidence", "description"), "reclassified": ""}
        cosmetic = entry["category"] in policy.COSMETIC_CATEGORIES
        if not cosmetic and entry["reference"] and entry["annotated"] and \
                policy.canonical_label(entry["reference"]) == policy.canonical_label(entry["annotated"]):
            entry["reclassified"], entry["category"], cosmetic = "equivalent labels", "label_format", True
        if not cosmetic and entry["category"] in ("", "other_chemical"):
            notation, structural = policy.classify_text(" ".join((entry["reference"], entry["annotated"], entry["detail"])))
            if notation and not structural:
                entry["reclassified"], entry["category"], cosmetic = "notation wording", sorted(notation)[0], True
        entry["category"] = entry["category"] or "other_chemical"
        entry["cosmetic"] = cosmetic
        out.append(entry)
    return out


def judge_structured(rec: dict) -> Judgement:
    """A call made with the policy prompt. Re-derives everything from the model's own fields; the parser's
    `decision` is deliberately not used, so a contradictory ACCEPT can be flagged here instead of vanishing."""
    idx = rec.get("run_idx")
    if rec.get("status") != "ok":
        return Judgement("INVALID", flags=[f"call_{rec.get('status') or 'failed'}"], run_idx=idx)
    data = rec.get("data")
    if not isinstance(data, dict):
        data, _ = extract_json_object(rec.get("raw_text") or "")
        data = {str(k).strip().lower(): v for k, v in (data or {}).items()}
    if not data:
        return Judgement("INVALID", flags=["unparseable_output"], run_idx=idx)
    verdict = str(data.get("verdict", "")).strip().upper()
    if verdict not in (ACCEPT, REJECT, UNCERTAIN):
        return Judgement("INVALID", flags=["missing_or_unknown_verdict"], run_idx=idx)
    equivalence = tri_state(data.get("chemical_equivalence"))
    checklist = _checklist(data.get("checklist"))
    discrepancies = _discrepancies(data.get("discrepancies"))
    chemical = [d for d in discrepancies if not d["cosmetic"]]
    supported = [d for d in chemical if d["location"] and (d["reference"] or d["annotated"] or d["detail"])]
    notation = sorted({d["category"] for d in discrepancies if d["cosmetic"]} | {
        policy.normalise_category(_text(n, "category")) for n in data.get("notation_differences") or [] if isinstance(n, dict)} - {""})
    unreadable = bool(data.get("unreadable_regions")) or "unreadable" in checklist.values()
    j = Judgement("INVALID", notation=notation, confidence=_confidence(data.get("confidence")), run_idx=idx)
    flags = j.flags
    if verdict == ACCEPT:
        if equivalence is not True:
            flags.append("accept_equivalence_conflict")
        elif supported or any(s == "mismatch" for s in checklist.values()):
            flags.append("accept_with_discrepancy")
        elif unreadable:
            j.vote = UNCERTAIN
            flags.append("accept_with_unreadable_region")
        elif data.get("requires_review") is True:
            j.vote = UNCERTAIN
            flags.append("accept_requires_review")
        else:
            j.vote = ACCEPT
            complete = all(checklist.get(a) == "match" or (a in policy.OPTIONAL_ASPECTS and checklist.get(a) == "not_present")
                           for a in policy.CHECKLIST_ASPECTS)
            j.verified = complete
            if not complete:
                flags.append("accept_without_complete_checklist")
    elif verdict == REJECT:
        if supported:
            if equivalence is True or (checklist and all(s == "match" for s in checklist.values())):
                j.vote = UNCERTAIN
                flags.append("reject_contradicts_equivalence")
            else:
                j.vote, j.evidence = REJECT, [{k: d[k] for k in ("category", "location", "reference", "annotated", "detail")} for d in supported]
        elif unreadable:
            j.vote = UNCERTAIN
            flags.append("reject_with_unreadable_region")
        else:
            j.vote = "EXCLUDED"
            flags.append("unsupported_reject_ignored")
            flags.append("reject_notation_only" if discrepancies or notation else "reject_without_evidence")
            if chemical:
                flags.append("reject_missing_location")
    else:  # UNCERTAIN
        j.vote = UNCERTAIN
        if not unreadable:
            flags.append("uncertain_without_unreadable_region")
    return j


def judge_confirmation(rec: dict) -> Confirmation:
    """A confirming call (policy.confirmation_prompt). Re-derives the status from the model's own fields, so a 'confirmed'
    with missing or identical observations, or with unreadable regions, never counts as a confirmation."""
    key, idx = str(rec.get("claim_key") or ""), rec.get("run_idx")
    if rec.get("status") != "ok":
        return Confirmation("invalid", key, flags=[f"call_{rec.get('status') or 'failed'}"], run_idx=idx)
    data = rec.get("data")
    if not isinstance(data, dict):
        data, _ = extract_json_object(rec.get("raw_text") or "")
        data = {str(k).strip().lower(): v for k, v in (data or {}).items()}
    status = str((data or {}).get("claim_status", "")).strip().lower()
    if status not in policy.CONFIRM_STATUSES:
        return Confirmation("invalid", key, flags=["missing_or_unknown_claim_status"], run_idx=idx)
    c = Confirmation(status, key, {"reference": _text(data, "image_a_observation"), "annotated": _text(data, "image_b_observation")},
                     [], _confidence(data.get("confidence")), _text(data, "specific_evidence"), idx)
    if status == "confirmed":
        problems = claim_problems(data)
        if problems:
            c.status = "invalid"
            c.flags.append("confirmed_claim_invalid")
            c.statement = c.statement or "; ".join(problems)
        elif data.get("unreadable_regions"):
            c.status = "inconclusive"
            c.flags.append("confirmed_with_unreadable_region")
    return c


# --------------------------------------------------------------------------- judgements -> decision


def _claim_dict(evidence: list) -> dict | None:
    if not evidence:
        return None
    top = evidence[0]
    return {k: top[k] for k in ("category", "location", "reference", "annotated", "detail", "votes")}


def decide(judgements: list[Judgement], confirmation: Confirmation | None = None, graph: GraphResult = NOT_PROVIDED, *,
           quorum: int = policy.QUORUM, max_structured: int = policy.MAX_STRUCTURED_CALLS, confirmation_enabled: bool = True) -> Decision:
    verified = sum(1 for j in judgements if j.vote == ACCEPT and j.verified)
    supported = [j for j in judgements if j.vote == REJECT]
    uncertain = [j for j in judgements if j.vote == UNCERTAIN]
    graph_state = graph.status if graph.status in ("match", "mismatch") else ""
    calls_left = len(judgements) < max_structured

    merged: dict[str, dict] = {}
    for j in supported:
        for e in j.evidence:
            slot = merged.setdefault(claim_key(e["category"], e["location"]), {**e, "votes": 0})
            slot["votes"] += 1
    evidence = sorted(merged.values(), key=lambda e: (-e["votes"], e["category"], " ".join(e["location"].lower().split())))
    claim = _claim_dict(evidence)
    applicable = confirmation if (confirmation is not None and claim is not None
                                  and confirmation.claim_key == claim_key(claim["category"], claim["location"])) else None

    causes: list[str] = []
    escalate = need_confirmation = False
    evidence_status = "none"
    if graph_state == "mismatch":
        if supported:
            outcome, rule, evidence_status = REJECT, "graph_confirmed_reject", "graph_confirmed"
        else:
            outcome, rule, escalate = UNCERTAIN, "graph_vlm_conflict", not judgements
            causes.append("graph_conflict")
    elif supported:
        evidence_status = "asserted"
        if verified or graph_state == "match":
            outcome, rule, escalate = UNCERTAIN, "contested_reject", calls_left
            causes.append("contested_reject")
        elif applicable is not None:
            if applicable.status == "confirmed":
                outcome, rule, evidence_status = REJECT, "confirmed_reject", "confirmed"
            elif applicable.status == "refuted":
                outcome, rule, evidence_status = UNCERTAIN, "checks_disagree", "refuted"
                causes.append("checks_disagree")
            else:
                outcome, rule, evidence_status = UNCERTAIN, "reject_not_confirmed", "inconclusive"
                causes.append("reject_not_confirmed")
        elif confirmation_enabled:
            outcome, rule, need_confirmation = UNCERTAIN, "reject_pending_confirmation", True
            causes.append("reject_pending_confirmation")
        else:
            outcome, rule = UNCERTAIN, "confirmation_not_run"
            causes.append("confirmation_not_run")
    elif uncertain:
        outcome, rule, escalate = UNCERTAIN, "unreadable_or_uncertain", calls_left
        causes.append("unreadable_region")
    elif verified + (1 if graph_state == "match" else 0) >= quorum and verified >= 1:
        outcome, rule = ACCEPT, "verified_equivalence"
    else:
        outcome, rule, escalate = UNCERTAIN, "insufficient_evidence", calls_left
        causes.append("insufficient_evidence")
    if outcome == UNCERTAIN:
        if rule in ("contested_reject", "unreadable_or_uncertain", "insufficient_evidence") and not escalate and not calls_left:
            causes.append("escalation_exhausted")
        if any(j.vote == "INVALID" for j in judgements):
            causes.append("malformed_output")
        if not (supported or verified) and any(j.vote == "EXCLUDED" for j in judgements):
            causes.append("unsupported_reject_only")

    if graph_state == "mismatch":
        evidence.append({"category": ",".join(graph.categories) or "graph_mismatch", "location": "whole structure", "reference": "",
                         "annotated": "", "detail": graph.detail, "votes": 1})
    failure = sorted({e["category"] for e in evidence}) if outcome == REJECT else (sorted(set(causes)) if outcome == UNCERTAIN else [])
    flags = {f for j in judgements for f in j.flags}
    if applicable is not None:
        flags.update(applicable.flags)
    if confirmation is not None and claim is not None and applicable is None:
        flags.add("stale_confirmation_ignored")
    votes = Counter(j.vote for j in judgements)
    matching = [j for j in judgements if j.vote == outcome]
    confidences = [j.confidence for j in matching if j.confidence is not None]
    return Decision(
        outcome=outcome, rule=rule, failure_categories=failure, evidence=evidence,
        notation_differences=sorted({n for j in judgements for n in j.notation}), flags=sorted(flags),
        needs_escalation=escalate, needs_confirmation=need_confirmation, requires_review=outcome == UNCERTAIN,
        votes={v: votes.get(v, 0) for v in VOTES},
        accept_share=round(verified / len(judgements), 2) if judgements else None,
        model_confidence=round(sum(confidences) / len(confidences), 2) if confidences else None,
        structured_calls=len(judgements), verified_accepts=verified, graph=graph.label(), claim=claim, evidence_status=evidence_status,
        confirmation=None if applicable is None else {"status": applicable.status, "reference": applicable.observations.get("reference", ""),
                                                      "annotated": applicable.observations.get("annotated", ""),
                                                      "confidence": applicable.confidence, "run_idx": applicable.run_idx},
        calls=[{"run_idx": j.run_idx, "vote": j.vote, "verified": j.verified, "confidence": j.confidence, "flags": j.flags} for j in judgements],
    )


# --------------------------------------------------------------------------- explanation and consistency gate


def _claim_text(claim: dict | None) -> str:
    if not claim:
        return "a difference"
    return (f"{claim['category']} at \"{claim['location']}\" (Image A: {claim.get('reference') or 'not stated'}; "
            f"Image B: {claim.get('annotated') or 'not stated'})")


def explain(d: Decision) -> str:
    """One deterministic sentence group written from the final decision only: it can say nothing the decision does not hold."""
    notation = f" Notation differences noted, not counted as errors: {', '.join(d.notation_differences)}." if d.notation_differences else ""
    n = d.structured_calls
    if d.outcome == ACCEPT:
        graph = f" Graph comparison: {d.graph}." if d.graph.startswith("match") else ""
        return (f"ACCEPT ({d.rule}): {d.verified_accepts} of {n} policy-prompt call(s) judged the drawings chemically equivalent with every "
                f"checklist item matching, and none reported a supported chemical discrepancy.{graph}{notation} "
                "Model-assessed from the images; not verified by an independent structural tool.")
    if d.outcome == REJECT:
        text = f"REJECT ({d.rule}): reported {_claim_text(d.claim)}, cited by {d.claim['votes']} of {n} policy-prompt call(s)."
        if d.evidence_status == "confirmed" and d.confirmation:
            text += (f" A separate confirming call (different prompt) observed Image A: {d.confirmation['reference'] or 'not stated'}; "
                     f"Image B: {d.confirmation['annotated'] or 'not stated'}, and confirmed that the difference is chemical. "
                     "Confirmed by a second Gemini check, not by an independent structural tool.")
        elif d.evidence_status == "graph_confirmed":
            text += f" A graph comparison also reported a mismatch ({d.graph})."
        return text + notation
    causes = {
        "contested_reject": lambda: f"a policy call reported {_claim_text(d.claim)}, but {d.verified_accepts} call(s) judged the drawings equivalent and the disagreement was not resolved",
        "reject_pending_confirmation": lambda: f"a reported difference ({_claim_text(d.claim)}) is waiting for its confirming call",
        "confirmation_not_run": lambda: f"a policy call reported {_claim_text(d.claim)}, but the confirming call was not run, so the claim is unconfirmed",
        "checks_disagree": lambda: (f"a policy call reported {_claim_text(d.claim)}, but the confirming call did not confirm it "
                                    f"(Image A: {(d.confirmation or {}).get('reference') or 'not stated'}; Image B: {(d.confirmation or {}).get('annotated') or 'not stated'})"),
        "reject_not_confirmed": lambda: f"a policy call reported {_claim_text(d.claim)}, but the confirming call was inconclusive or invalid",
        "unreadable_or_uncertain": lambda: "at least one call could not read a critical part of a drawing",
        "insufficient_evidence": lambda: f"{d.verified_accepts} of the required {policy.QUORUM} clean accepts were recorded",
        "graph_vlm_conflict": lambda: f"the graph comparison reported a mismatch ({d.graph}) that no policy call supports",
        "consistency_conflict": lambda: f"the consistency check found: {', '.join(d.consistency)}",
    }
    reason = causes.get(d.rule, lambda: "the evidence does not support ACCEPT or REJECT")()
    return f"UNCERTAIN ({d.rule}): {reason}.{notation} Review required."


def check_consistency(d: Decision) -> list[str]:
    """Contradictions between outcome, evidence, explanation and review flag. Empty list = consistent."""
    v: list[str] = []
    if sum(d.votes.values()) != d.structured_calls:
        v.append("vote_count_mismatch")
    if d.outcome == ACCEPT:
        if d.evidence or d.failure_categories:
            v.append("accept_with_evidence")
        if d.requires_review:
            v.append("accept_flagged_for_review")
        if "REJECT" in d.explanation:
            v.append("explanation_contradicts_outcome")
    elif d.outcome == REJECT:
        top = d.evidence[0] if d.evidence else None
        if d.evidence_status not in ("confirmed", "graph_confirmed"):
            v.append("reject_unconfirmed")
        if top is None or top["category"] not in policy.CHEMICAL_CATEGORIES and d.evidence_status != "graph_confirmed" \
                or not top["location"] or not (top["reference"] or top["annotated"] or top["detail"]):
            v.append("reject_without_complete_evidence")
        if d.evidence_status == "confirmed" and d.verified_accepts:
            v.append("reject_with_verified_accept")
        if d.evidence_status == "confirmed" and not (d.confirmation and d.confirmation.get("status") == "confirmed"):
            v.append("reject_without_confirmation_record")
        if d.requires_review:
            v.append("reject_flagged_for_review")
        if d.claim and not (d.claim["category"] in d.explanation and d.claim["location"] in d.explanation):
            v.append("explanation_missing_evidence")
    else:
        if not d.requires_review:
            v.append("uncertain_not_flagged")
        if not d.failure_categories:
            v.append("uncertain_without_cause")
        if d.needs_confirmation:
            v.append("unconfirmed_pending_at_export")
    if not d.explanation.startswith(d.outcome):
        v.append("explanation_outcome_mismatch")
    return v


def finalize(d: Decision) -> Decision:
    """Write the explanation, run the gate, and refuse to publish a contradictory row as a clean ACCEPT or REJECT."""
    d.explanation = explain(d)
    violations = check_consistency(d)
    if violations:
        d.consistency, d.consistency_status = violations, "conflict"
        d.outcome, d.rule, d.requires_review = UNCERTAIN, "consistency_conflict", True
        d.failure_categories = sorted({"consistency_conflict", *violations})
        d.needs_escalation = d.needs_confirmation = False
        d.explanation = explain(d)
    else:
        d.consistency_status = "ok"
    return d


def publish(judgements: list[Judgement], confirmation: Confirmation | None = None, graph: GraphResult = NOT_PROVIDED, *,
            final: bool = True, **kwargs) -> Decision:
    """The decision as it may be exported: rules, explanation and gate. `final=True` means no further call will be made,
    so a reject still waiting for its confirming call is exported as UNCERTAIN confirmation_not_run."""
    return finalize(decide(judgements, confirmation, graph, confirmation_enabled=not final, **kwargs))


def votes_text(votes: dict) -> str:
    return ";".join(f"{k}:{votes[k]}" for k in VOTES)


def evidence_json(evidence: list) -> str:
    return json.dumps(evidence, ensure_ascii=False, separators=(",", ":")) if evidence else ""
