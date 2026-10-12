"""Turn raw run records into the accuracy numbers that gate a prompt.

Fail-closed scoring: only an exact decision of 1 is an accept. 'unsure' (2), malformed output,
blocked or empty responses and API errors are never accepts. A wrong drawing counts as falsely
accepted if ANY of its runs accepted it. Runs on one pair are correlated, so intervals use pairs
as the unit (exact binomial) or a cluster bootstrap over pairs.

Headline numbers use only the annotator pairs (a molecule's wrong and corrected drawing against its
reference). Cross-molecule pairs are trivially easy negatives; mixing them in would make the
false-accept bound look tighter than the real task supports, so they are reported separately.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Callable

from .stats import clopper_pearson, clopper_pearson_upper, ratio_bootstrap

OUTCOMES = ("true_accept", "true_reject", "false_accept", "false_reject", "unsure", "invalid", "blocked", "error")
CORRECT = frozenset({"true_accept", "true_reject"})
ANNOTATED = ("annotated_wrong", "annotated_correct")
# The model answered, but with nothing usable (safety block, empty text). Unlike a transient API
# error this is a verdict on that image, so it stays in the denominators and is never an accept.
BLOCKED_STATUSES = frozenset({"blocked", "empty"})


def outcome(rec: dict) -> str:
    status = rec.get("status")
    if status in BLOCKED_STATUSES:
        return "blocked"
    if status != "ok":
        return "error"
    decision = rec.get("decision")
    if decision is None:
        return "invalid"
    if decision == 2:
        return "unsure"
    if rec["expected"] == 0:
        return "false_accept" if decision == 1 else "true_reject"
    return "true_accept" if decision == 1 else "false_reject"


def pair_details(records) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        grouped[rec["task_id"]].append(rec)
    details = []
    for pair_id, recs in sorted(grouped.items()):
        recs.sort(key=lambda r: r["run_idx"])
        first = recs[0]
        details.append(
            {
                "pair_id": pair_id,
                "molecule": first["molecule"],
                "b_molecule": first.get("b_molecule"),
                "category": first["category"],
                "expected": first["expected"],
                "gated": first["gated"],
                "outcomes": [outcome(r) for r in recs],
                "decisions": [r.get("decision") if r.get("status") == "ok" else None for r in recs],
                "reasons": [r.get("reason", "") for r in recs],
            }
        )
    return details


def _clusters(items: list[dict], wanted: set[str] | frozenset[str]) -> list[tuple[int, int]]:
    """(hits, answered runs) per pair; API-error runs are excluded from the denominator."""
    return [
        (sum(o in wanted for o in d["outcomes"]), sum(o != "error" for o in d["outcomes"]))
        for d in items
    ]


def _rate(clusters: list[tuple[int, int]], n_boot: int, seed: int) -> dict:
    num = sum(c[0] for c in clusters)
    den = sum(c[1] for c in clusters)
    lo, hi = ratio_bootstrap(clusters, n_boot, seed)
    return {"num": num, "den": den, "rate": (num / den if den else None), "ci95": [lo, hi] if lo is not None else None}


def summarise(items: list[dict], *, n_boot: int = 2000, seed: int = 0) -> dict:
    runs: Counter = Counter()
    for d in items:
        runs.update(d["outcomes"])
    answered = sum(runs.values()) - runs["error"]
    neg = [d for d in items if d["expected"] == 0 and any(o != "error" for o in d["outcomes"])]
    pos = [d for d in items if d["expected"] == 1 and any(o != "error" for o in d["outcomes"])]
    fa_ids = [d["pair_id"] for d in neg if "false_accept" in d["outcomes"]]
    fr_ids = [d["pair_id"] for d in pos if "false_reject" in d["outcomes"]]

    decided = runs["true_accept"] + runs["true_reject"] + runs["false_accept"] + runs["false_reject"]
    agreements = []
    for d in items:
        values = [dec for dec, o in zip(d["decisions"], d["outcomes"]) if o != "error"]
        if len(values) >= 2:
            agreements.append(max(Counter(values).values()) / len(values))
    all_correct = [
        d for d in items
        if any(o != "error" for o in d["outcomes"]) and all(o in CORRECT or o == "error" for o in d["outcomes"])
    ]
    evaluated = [d for d in items if any(o != "error" for o in d["outcomes"])]

    return {
        "n_pairs": len(items),
        "n_runs": sum(runs.values()),
        "api_errors": runs["error"],
        "run_outcomes": {k: runs[k] for k in OUTCOMES},
        "negatives": {  # drawings that should be rejected
            "n_pairs": len(neg),
            "false_accept_pairs": len(fa_ids),
            "false_accept_ids": fa_ids,
            "pair_rate_ci95": list(clopper_pearson(len(fa_ids), len(neg))) if neg else None,
            "upper95_one_sided": clopper_pearson_upper(len(fa_ids), len(neg)) if neg else None,
            "false_accept_runs": _rate(_clusters(neg, {"false_accept"}), n_boot, seed),
        },
        "positives": {  # drawings that should be accepted
            "n_pairs": len(pos),
            "false_reject_pairs": len(fr_ids),
            "false_reject_ids": fr_ids,
            "pair_rate_ci95": list(clopper_pearson(len(fr_ids), len(pos))) if pos else None,
            "false_reject_runs": _rate(_clusters(pos, {"false_reject"}), n_boot, seed),
        },
        "unsure_runs": _rate(_clusters(items, {"unsure"}), n_boot, seed),
        "invalid_runs": _rate(_clusters(items, {"invalid"}), n_boot, seed),
        "blocked_runs": _rate(_clusters(items, {"blocked"}), n_boot, seed),
        "strict_accuracy_runs": _rate(_clusters(items, CORRECT), n_boot, seed),
        "decided_accuracy": ((runs["true_accept"] + runs["true_reject"]) / decided) if decided else None,
        "coverage": (decided / answered) if answered else None,
        "pair_all_correct": {"num": len(all_correct), "den": len(evaluated)},
        "stability": {
            "pairs": len(agreements),
            "mean_agreement": (sum(agreements) / len(agreements)) if agreements else None,
            "unanimous_pairs": sum(a == 1.0 for a in agreements),
        },
    }


def flagged_correct(details: list[dict], issues: dict[str, list[dict]], *, kw: dict) -> dict:
    """Corrected drawings that the gold labelling says still have an issue.

    The CSV calls them correct, so they are still expected to be accepted. But if gold is right, a
    run that accepted one is a possible false accept and a run that rejected one may be justified.
    """
    correct = [d for d in details if d["category"] == "annotated_correct"]
    flagged = [d for d in correct if issues.get(d["molecule"])]
    clean = [d for d in correct if not issues.get(d["molecule"])]
    return {
        "n_correct_pairs": len(correct),
        "n_flagged": len(flagged),
        "accepted_by_any": [d["pair_id"] for d in flagged if "true_accept" in d["outcomes"]],
        "rejected_by_any": [d["pair_id"] for d in flagged if "false_reject" in d["outcomes"]],
        "pairs": [{**d, "issues": issues[d["molecule"]]} for d in flagged],
        "unflagged": summarise(clean, **kw)["positives"],
    }


def vote_rules(items: list[dict]) -> dict:
    """What each production rule 'accept only if at least k of N calls say 1' would have done.

    N is the number of calls made per pair in this run, so the rule must be run with the N that
    production will actually use. k = 1 is the loosest rule (any call accepts); k = N is unanimity.
    Calls that errored, were blocked, said 2 or were malformed count as 'not 1' (fail closed).
    """
    evaluated = [d for d in items if any(o != "error" for o in d["outcomes"])]
    n = max((len(d["outcomes"]) for d in evaluated), default=0)
    neg = [d for d in evaluated if d["expected"] == 0]
    pos = [d for d in evaluated if d["expected"] == 1]

    def accepts(d: dict) -> int:
        return sum(o in ("true_accept", "false_accept") for o in d["outcomes"])

    rows = [
        {
            "k": k,
            "false_accept_pairs": sum(accepts(d) >= k for d in neg),
            "false_reject_pairs": sum(accepts(d) < k for d in pos),
        }
        for k in range(1, n + 1)
    ]
    neg_runs = [o for d in neg for o in d["outcomes"] if o != "error"]
    pos_runs = [o for d in pos for o in d["outcomes"] if o != "error"]
    single = {  # one call per drawing: every answered run is its own decision
        "false_accept_runs": sum(o == "false_accept" for o in neg_runs),
        "n_negative_runs": len(neg_runs),
        "sent_back_runs": sum(o != "true_accept" for o in pos_runs),
        "n_positive_runs": len(pos_runs),
    }
    return {"n_calls": n, "n_negatives": len(neg), "n_positives": len(pos), "single_call": single, "rules": rows}


def pairs_needed(max_rate: float, confidence: float = 0.95) -> int:
    """Wrong drawings that must ALL be rejected before 'true false-accept rate below max_rate' holds."""
    return math.ceil(math.log(1 - confidence) / math.log(1 - max_rate))


def headline(result: dict) -> tuple[dict, list[dict], str]:
    """(summary, pairs, basis) that the gate view should show: annotator pairs when there are any."""
    if result["annotated"]["n_pairs"]:
        return result["annotated"], [d for d in result["pairs"] if d["gated"] and d["category"] in ANNOTATED], "annotated"
    return result["gated"], [d for d in result["pairs"] if d["gated"]], "gated"


def compute(
    records,
    *,
    split_of: Callable[[str], str] | None = None,
    gold_primary: dict[str, str] | None = None,
    correct_issues: dict[str, list[dict]] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    details = pair_details(records)
    kw = {"n_boot": n_boot, "seed": seed}
    gated = [d for d in details if d["gated"]]
    annotated = [d for d in gated if d["category"] in ANNOTATED]
    result: dict = {
        "overall": summarise(details, **kw),
        "gated": summarise(gated, **kw),
        "annotated": summarise(annotated, **kw),
        "by_category": {c: summarise([d for d in details if d["category"] == c], **kw) for c in sorted({d["category"] for d in details})},
        "by_split": {},
        "by_gold_error": {},
        "flagged_correct": flagged_correct(details, correct_issues, kw=kw) if correct_issues else None,
        "pairs": details,
    }
    core = annotated or gated
    result["vote_rules"] = vote_rules(core)
    if split_of:
        for name in sorted({split_of(d["molecule"]) for d in core}):
            result["by_split"][name] = summarise([d for d in core if split_of(d["molecule"]) == name], **kw)
    if gold_primary:
        wrong = [d for d in details if d["category"] == "annotated_wrong"]
        for name in sorted({gold_primary.get(d["molecule"], "unknown") for d in wrong}):
            result["by_gold_error"][name] = summarise([d for d in wrong if gold_primary.get(d["molecule"], "unknown") == name], **kw)
    return result
