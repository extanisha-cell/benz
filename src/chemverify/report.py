"""Markdown reports for a run, and side-by-side comparison of runs."""
from __future__ import annotations

from collections import Counter
from typing import Callable

from .metrics import headline, pairs_needed


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def interval(c) -> str:
    return "" if not c or c[0] is None else f"[{pct(c[0])}, {pct(c[1])}]"


def _rate_cell(rate: dict) -> str:
    if not rate or rate.get("den") in (0, None):
        return "n/a"
    return f"{rate['num']}/{rate['den']} = {pct(rate['rate'])} {interval(rate['ci95'])}".strip()


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def _confusion(pairs: list[dict]) -> list[str]:
    cols = ["0 (reject)", "1 (accept)", "2 (cannot verify)", "malformed", "blocked / empty", "API error"]
    key = {
        0: {"true_reject": 0, "false_accept": 1, "unsure": 2, "invalid": 3, "blocked": 4, "error": 5},
        1: {"false_reject": 0, "true_accept": 1, "unsure": 2, "invalid": 3, "blocked": 4, "error": 5},
    }
    rows = []
    for expected, label in ((0, "should reject"), (1, "should accept")):
        counts = [0] * len(cols)
        for d in pairs:
            if d["expected"] == expected:
                for o in d["outcomes"]:
                    counts[key[expected][o]] += 1
        rows.append([label] + [str(c) for c in counts])
    return _table(["expected \\ model output"] + cols, rows)


def _group_table(groups: dict[str, dict]) -> list[str]:
    rows = []
    for name, s in groups.items():
        neg, pos = s["negatives"], s["positives"]
        rows.append([
            name, str(s["n_pairs"]), str(s["n_runs"]),
            f"{neg['false_accept_pairs']}/{neg['n_pairs']}" if neg["n_pairs"] else "-",
            f"{pos['false_reject_pairs']}/{pos['n_pairs']}" if pos["n_pairs"] else "-",
            pct(s["unsure_runs"]["rate"]), pct(s["strict_accuracy_runs"]["rate"]),
        ])
    return _table(["group", "pairs", "runs", "wrong accepted (pairs)", "correct rejected (pairs)", "unsure runs", "strict accuracy"], rows)


def _sample_size_note(n_negatives: int, false_accepts: int) -> str:
    needs = ", ".join(f"below {pct(p)}: {pairs_needed(p)}" for p in (0.10, 0.05, 0.02))
    text = (f"How many wrong drawings it takes to certify a false-accept rate (95% confidence, zero false accepts allowed): {needs}. "
            f"This run has {n_negatives}")
    return text + ("." if not false_accepts else f", and {false_accepts} of them were accepted, so those targets are out of reach until that is fixed.")


def _vote_section(vr: dict) -> list[str]:
    L = ["## Production rule: how many calls per drawing, and how many must say 1", ""]
    if not vr["n_calls"]:
        return L + ["No answered runs.", ""]
    n = vr["n_calls"]
    s = vr["single_call"]
    L.append(f"This run made {n} call(s) per pair. A rule only applies if production makes the same number of calls per drawing; "
             "each extra call costs money and time. Errors, 2, malformed and blocked answers all count as 'not 1'.")
    L.append("")
    rows = [["1 call per drawing (each run on its own)", f"{s['false_accept_runs']}/{s['n_negative_runs']} runs",
             f"{s['sent_back_runs']}/{s['n_positive_runs']} runs", "1"]]
    for r in vr["rules"]:
        k = r["k"]
        name = f"any of {n} says 1" if k == 1 and n > 1 else (f"all {n} say 1" if k == n and n > 1 else f"at least {k} of {n} say 1")
        if n == 1:
            continue  # identical to the single-call row
        rows.append([name, f"{r['false_accept_pairs']}/{vr['n_negatives']} pairs", f"{r['false_reject_pairs']}/{vr['n_positives']} pairs", str(n)])
    L += _table(["Rule", "Wrong drawings accepted", "Correct drawings sent back", "Calls per drawing"], rows)
    L.append("")
    return L


def _flagged_section(fc: dict) -> list[str]:
    L = ["## Corrected drawings that gold labelling says still have an issue", ""]
    if not fc["n_flagged"]:
        return L + [f"None of the {fc['n_correct_pairs']} corrected drawings has a recorded issue.", ""]
    L.append(
        f"{fc['n_flagged']} of {fc['n_correct_pairs']} corrected drawings have an issue recorded by the gold agents. "
        "The CSV calls them correct, so they are still counted as 'should accept' above. If the gold agents are right, "
        f"an accept here is a **possible false accept** ({len(fc['accepted_by_any'])} pair(s) accepted by at least one run) "
        f"and a reject may be justified ({len(fc['rejected_by_any'])} pair(s) rejected by at least one run). "
        "Check these against `data/gold/review_table.md`."
    )
    L.append("")
    rows = []
    for d in fc["pairs"]:
        issues = "; ".join(f"[{i['category']}, {i['severity']}] {i['description']}" for i in d["issues"])
        rows.append([d["pair_id"], issues[:200].replace("|", "/"), " ".join("-" if x is None else str(x) for x in d["decisions"])])
    L += _table(["pair", "issue(s) recorded in the corrected drawing", "decisions per run"], rows)
    unflagged = fc["unflagged"]
    L.append("")
    L.append(f"Corrected drawings with no recorded issue sent back: {unflagged['false_reject_pairs']}/{unflagged['n_pairs']} "
             f"(exact 95% CI {interval(unflagged['pair_rate_ci95'])}).")
    L.append("")
    return L


def render(
    meta: dict,
    metrics: dict,
    *,
    show_heldout: bool = False,
    split_of: Callable[[str], str] | None = None,
    localisation: dict[str, bool] | None = None,
    heldout_excluded: int = 0,
    model_versions: dict[str, int] | None = None,
    spend: dict | None = None,
) -> str:
    L: list[str] = []
    total = metrics["gated"]
    g, core_pairs, basis = headline(metrics)
    neg, pos = g["negatives"], g["positives"]
    settings = meta.get("settings", {})

    L.append(f"# Run report: {meta.get('name', '?')}")
    L.append("")
    L.append(f"- Prompt: `{meta.get('prompt_name')}` (sha `{meta.get('prompt_sha')}`), output format `{meta.get('format')}`")
    shown = {k: v for k, v in settings.items() if v is not None}
    L.append(f"- Model and settings (unset = API default): {shown}")
    if model_versions:
        L.append(f"- Model version(s) the API reported, with call counts: {model_versions}")
        if len(model_versions) > 1:
            L.append("- **WARNING: more than one model version answered this run, so its results mix versions. "
                     "Preview models can change under the same name; start a new run name to compare cleanly.**")
    if spend and spend.get("calls"):
        L.append(f"- Cost the API reported: ${spend['usd']:.2f} over {spend['calls']} calls (about ${spend['usd'] / spend['calls']:.3f} per call)")
    L.append(f"- Pairs: {total['n_pairs']} gated pairs ({g['n_pairs']} in the gate view), {total['n_runs']} runs in total")
    if heldout_excluded:
        L.append(f"- {heldout_excluded} held-out pair(s) are left out of every number in this report so they cannot steer prompt edits; "
                 "pass --show-heldout for the final test only.")
    if total["api_errors"]:
        L.append(f"- **WARNING: {total['api_errors']} runs failed with API errors and are excluded below. Re-run to complete them.**")
    if total["run_outcomes"]["blocked"]:
        L.append(f"- **WARNING: {total['run_outcomes']['blocked']} runs were blocked or came back empty. They count as non-accepts and stay in the denominators.**")
    L.append("")

    if basis == "annotated":
        L.append("## Gate view (annotator pairs: each molecule's wrong and corrected drawing vs its reference)")
        L.append("")
        L.append("Cross-molecule pairs are trivially easy negatives, so they are left out of the bound below and listed under 'By pair category'.")
    else:
        L.append("## Gate view (gated pairs; no annotator pairs in this run)")
    L.append("")
    fa_bound = neg["upper95_one_sided"]
    rows = [
        ["Wrong drawings accepted (a pair counts if ANY run accepted it)",
         f"{neg['false_accept_pairs']}/{neg['n_pairs']}",
         f"exact 95% CI {interval(neg['pair_rate_ci95'])}; true rate below {pct(fa_bound)} with 95% confidence"],
        ["Correct drawings sent back (a pair counts if ANY run rejected it)",
         f"{pos['false_reject_pairs']}/{pos['n_pairs']}", f"exact 95% CI {interval(pos['pair_rate_ci95'])}"],
        ["Wrong accepted, per run", _rate_cell(neg["false_accept_runs"]), "cluster bootstrap over pairs"],
        ["Correct rejected, per run", _rate_cell(pos["false_reject_runs"]), "cluster bootstrap over pairs"],
        ["'Cannot verify' (2), per run", _rate_cell(g["unsure_runs"]), ""],
        ["Malformed output, per run", _rate_cell(g["invalid_runs"]), "format compliance"],
        ["Blocked or empty response, per run", _rate_cell(g["blocked_runs"]), "counted as a non-accept, kept in the denominator"],
        ["Strict accuracy, per run (2, malformed and blocked count as wrong)", _rate_cell(g["strict_accuracy_runs"]), ""],
        ["Accuracy when a decision was given (0 or 1)", pct(g["decided_accuracy"]), f"coverage {pct(g['coverage'])}"],
        ["Stability (same decision across repeated runs)",
         f"{pct(g['stability']['mean_agreement'])} mean agreement",
         f"{g['stability']['unanimous_pairs']}/{g['stability']['pairs']} pairs unanimous"],
    ]
    cross = metrics["by_category"].get("cross_molecule")
    if basis == "annotated" and cross and cross["negatives"]["n_pairs"]:
        cn = cross["negatives"]
        rows.insert(2, ["Different compound accepted (cross-molecule sanity check)",
                        f"{cn['false_accept_pairs']}/{cn['n_pairs']}", "must be 0; says nothing about subtle errors"])
    L += _table(["Measure", "Result", "Interval / note"], rows)
    L.append("")
    L.append(_sample_size_note(neg["n_pairs"], neg["false_accept_pairs"]))
    L.append("")

    L.append(f"## Confusion matrix (runs, {'annotator' if basis == 'annotated' else 'gated'} pairs)")
    L.append("")
    L += _confusion(core_pairs)
    L.append("")

    if metrics.get("vote_rules"):
        L += _vote_section(metrics["vote_rules"])

    if metrics.get("flagged_correct"):
        L += _flagged_section(metrics["flagged_correct"])

    L.append("## By pair category")
    L.append("")
    L += _group_table(metrics["by_category"])
    L.append("")

    if metrics["by_split"]:
        L.append("## By split")
        L.append("")
        L += _group_table(metrics["by_split"])
        L.append("")
    if metrics["by_gold_error"]:
        L.append("## Wrong drawings by gold error category")
        L.append("")
        L += _group_table(metrics["by_gold_error"])
        L.append("")

    if localisation is not None:
        wrong = [d for d in metrics["pairs"] if d["category"] == "annotated_wrong" and "true_reject" in d["outcomes"]]
        judged = [d for d in wrong if d["pair_id"] in localisation]
        hits = sum(1 for d in judged if localisation[d["pair_id"]])
        L.append("## Localisation")
        L.append("")
        L.append(f"Rejected wrong drawings whose stated reason names the real error: {hits}/{len(judged)} judged ({len(wrong) - len(judged)} not yet judged).")
        L.append("")

    hidden = 0
    failing = []
    for d in metrics["pairs"]:
        if not d["gated"] or all(o in ("true_accept", "true_reject") for o in d["outcomes"]):
            continue
        if not show_heldout and split_of and split_of(d["molecule"]) == "heldout":
            hidden += 1
            continue
        reason = next((r for r in d["reasons"] if r), "")
        failing.append([d["pair_id"], str(d["expected"]), " ".join("-" if x is None else str(x) for x in d["decisions"]),
                        ", ".join(f"{k}x{v}" for k, v in sorted(Counter(d["outcomes"]).items())), reason[:160].replace("|", "/")])
    L.append("## Pairs that were not fully correct")
    L.append("")
    if failing:
        L += _table(["pair", "expected", "decisions per run", "outcomes", "first reason given"], failing)
    else:
        L.append("None.")
    if hidden:
        L.append("")
        L.append(f"_{hidden} held-out pair(s) with errors are hidden so they cannot steer prompt edits; pass --show-heldout for the final test only._")
    L.append("")
    L.append("**How to read this.** Decision 1 = accept, 0 = reject, 2 = cannot verify. Only an exact 1 is an accept. "
             "Zero false accepts on a small sample bounds the true rate but does not prove it is zero; the one-sided bound above shows how far the sample can certify.")
    return "\n".join(L) + "\n"


def _settings_cell(settings: dict) -> str:
    extra = {k: v for k, v in settings.items() if k != "model" and v is not None}
    return ", ".join(f"{k}={v}" for k, v in extra.items()) or "API defaults"


def render_compare(runs: list[tuple[dict, dict]]) -> str:
    rows = []
    for meta, metrics in runs:
        g = headline(metrics)[0]
        neg, pos = g["negatives"], g["positives"]
        rows.append([
            meta.get("name", "?"), meta.get("prompt_name", "?"), meta.get("settings", {}).get("model", "?"),
            _settings_cell(meta.get("settings", {})),
            str(g["n_pairs"]), f"{neg['false_accept_pairs']}/{neg['n_pairs']}", f"{pos['false_reject_pairs']}/{pos['n_pairs']}",
            pct(g["unsure_runs"]["rate"]), pct(g["invalid_runs"]["rate"]), pct(g["blocked_runs"]["rate"]),
            pct(g["strict_accuracy_runs"]["rate"]), pct(g["stability"]["mean_agreement"]),
        ])
    return "\n".join(_table(
        ["run", "prompt", "model", "other settings", "annotator pairs", "wrong accepted", "correct rejected", "cannot verify", "malformed", "blocked/empty",
         "strict accuracy", "stability"], rows
    )) + "\n"
