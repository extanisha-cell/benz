"""Verifier pipeline on the 10 unlabelled tasks in Google_Chem_10_tasks.csv.

CSV columns: IDs, Samples (typeset reference = Image A), Annotated (the annotator's drawing = Image B).
Nobody has said which annotated drawings are right, and there are no gold transcriptions, so this produces
verdicts, reasons and a perception cross-check, not accuracy numbers. Everything lives in data/google10/ and
runs/google10_*; the 17-molecule manifest, split and gold files are untouched (molecule_5 also exists in
data/tasks10, which is why the images are kept in a separate folder).

    python -I pipeline_google10.py prep --csv <Google_Chem_10_tasks.csv>
    python -I pipeline_google10.py verify --prompt prompts/baseline_v0.txt --name google10_verify --runs 1 \
        --thinking-level low --budget-usd 2 [--dry-run]
    python -I pipeline_google10.py probe --prompt prompts/perception_probe.txt --name google10_probe --runs 1 \
        --thinking-level low --budget-usd 1 [--dry-run]
    python -I pipeline_google10.py summary --verify runs/google10_verify --probe runs/google10_probe
    python -I pipeline_google10.py verify --prompt prompts/baseline_v0_conf.txt --name google10_verify_conf --runs 1 \
        --thinking-level low --budget-usd 1        # same task, but the model also states a 0-100 confidence
    python -I pipeline_google10.py csv --verify runs/google10_verify --probe runs/google10_probe \
        --verify-conf runs/google10_verify_conf

Another CSV: put `--set <name>` first (for example `--set google30 prep --csv Google_Chem_30_tasks.csv`). Images go to
data/<name>/, runs and outputs to runs/<name>_*, so the earlier dataset's files are never touched. Run names given
with --name are used as written.

Decision layer (chemverify.policy + chemverify.decision): one policy, deterministic rules, no model call unless asked.
    python -I pipeline_google10.py policy --write        # regenerate prompts/chem_equiv_policy_v1.txt from the policy
    python -I pipeline_google10.py decide --verify runs/google10_verify --probe runs/google10_probe \
        --verify-conf runs/google10_verify_conf          # reuses recorded responses; writes the results CSV (original
                                                         # columns first, decision columns appended); no model calls
    python -I pipeline_google10.py decide ... --escalate [--only molecule_25 ...] [--dry-run] [--budget-usd 1]
        # structured policy calls, only for tasks that are still unresolved (at most MAX_STRUCTURED_CALLS each)
    optional: --smiles-csv <csv with IDs, Samples_SMILES, Annotated_SMILES> turns on the RDKit graph check
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from chemverify import cli, decision, graphcheck, perception, policy, runner  # noqa: E402
from chemverify.download import DownloadError, atomic_write, urllib_fetch  # noqa: E402
from chemverify.imageinfo import sniff  # noqa: E402
from chemverify.manifest import ALLOWED_HOST_SUFFIXES, ID_RE, check_host, download_url, sanitize_url  # noqa: E402
from chemverify.pairs import Pair  # noqa: E402
from chemverify.prompts import load_prompt  # noqa: E402

SET = "google10"  # dataset name; `--set google30` (first argument) keeps another CSV's data and outputs apart
DATA10 = ROOT / "data" / SET
MANIFEST10 = DATA10 / "manifest.json"


def use_dataset(name: str) -> None:
    global SET, DATA10, MANIFEST10
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,30}", name):
        cli._die(f"--set must be lower-case letters, digits and underscores, got {name!r}")
    SET, DATA10 = name, ROOT / "data" / name
    MANIFEST10 = DATA10 / "manifest.json"


def run_file(suffix: str) -> str:
    """runs/<dataset>_<suffix>, the default location of every output of this dataset."""
    return str(ROOT / "runs" / f"{SET}_{suffix}")
ROLES10 = ("ref", "annotated")
COLUMNS = {"ids": "id", "samples": "ref", "annotated": "annotated"}
VERDICT_WORDS = {1: "accept", 0: "reject", 2: "cannot verify"}


def read_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        mapping = {n: COLUMNS[n.strip().lower()] for n in (reader.fieldnames or []) if n and n.strip().lower() in COLUMNS}
        missing = sorted(set(COLUMNS.values()) - set(mapping.values()))
        if missing:
            cli._die(f"CSV is missing columns for {missing}; expected headers {list(COLUMNS)}")
        rows, seen = [], set()
        for line_no, raw in enumerate(reader, start=2):
            vals = {mapping[k]: (v or "").strip() for k, v in raw.items() if k in mapping}
            if not any(vals.values()):
                continue
            if not ID_RE.match(vals["id"]) or vals["id"] in seen:
                cli._die(f"line {line_no}: unsafe, empty or duplicate id {vals['id']!r}")
            seen.add(vals["id"])
            for role in ROLES10:
                check_host(vals[role], ALLOWED_HOST_SUFFIXES)
            rows.append(vals)
    return rows


def fetch_image(url: str, folder: Path, role: str, force: bool) -> dict:
    """Download one image (or reuse a complete earlier download) and return its manifest entry."""
    folder.mkdir(parents=True, exist_ok=True)
    if not force:
        for existing in sorted(folder.glob(f"{role}.*")):
            data = existing.read_bytes()
            info = sniff(data)
            if info and info.complete:
                return entry_for(existing, data, info, url)
    data, content_type = urllib_fetch(download_url(url))
    if "text/html" in content_type.lower():
        raise DownloadError("received an HTML page instead of an image (file not shared, or needs confirmation)")
    info = sniff(data)
    if info is None:
        raise DownloadError("downloaded bytes are not a recognised image")
    for stale in folder.glob(f"{role}.*"):
        stale.unlink()
    path = folder / f"{role}.{info.extension}"
    atomic_write(path, data)
    return entry_for(path, data, info, url)


def entry_for(path: Path, data: bytes, info, url: str) -> dict:
    import hashlib
    return {"file": path.relative_to(DATA10).as_posix(), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            **info.as_dict(), "source": sanitize_url(url)}


def cmd_prep(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="prep")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    rows = read_rows(args.csv)
    molecules, problems = [], []
    for row in rows:
        images = {}
        for role in ROLES10:
            try:
                images[role] = fetch_image(row[role], DATA10 / "images" / row["id"], role, args.force)
                print(f"ok    {row['id']}/{role}", file=sys.stderr, flush=True)
            except Exception as exc:  # report and carry on
                problems.append(f"{row['id']}/{role}: {exc}")
                print(f"FAIL  {row['id']}/{role}: {exc}", file=sys.stderr, flush=True)
        molecules.append({"id": row["id"], "images": images, "complete": len(images) == len(ROLES10)})
    MANIFEST10.write_text(json.dumps({"version": 1, "molecules": molecules}, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'molecule':14s} {'role':10s} {'format':6s} {'size':>11s} {'bytes':>8s}  notes")
    for mol in molecules:
        for role, e in mol["images"].items():
            notes = [n for n, bad in (("has transparency", e["has_alpha"]), ("looks truncated", not e["complete"]),
                                      ("NOT accepted by Gemini", not e["gemini_supported"])) if bad]
            size = f"{e['width']}x{e['height']}" if e["width"] else "?"
            print(f"{mol['id']:14s} {role:10s} {e['format']:6s} {size:>11s} {e['bytes']:>8d}  {', '.join(notes)}")
    print(f"\n{sum(m['complete'] for m in molecules)}/{len(molecules)} molecules complete; manifest: {MANIFEST10}")
    for p in problems:
        print(f"  problem: {p}")
    return 1 if problems else 0


def load_manifest() -> list[dict]:
    if not MANIFEST10.exists():
        cli._die(f"{MANIFEST10} not found; run `prep` first")
    mols = json.loads(MANIFEST10.read_text(encoding="utf-8-sig"))["molecules"]
    complete = [m for m in mols if m["complete"]]
    if len(complete) != len(mols):
        cli._err(f"warning: {len(mols) - len(complete)} molecules are incomplete and are skipped")
    return complete


def pair_task_list(mols: list[dict], *, labels: bool = True, only=None) -> list:
    """One reference-vs-annotated task per molecule (no expected answer: nobody has labelled these)."""
    pairs = [Pair(f"{m['id']}__annotated", m["id"], m["id"], m["images"]["ref"]["file"], m["images"]["annotated"]["file"],
                  m["images"]["ref"]["sha256"], m["images"]["annotated"]["sha256"], None, "unlabelled", False) for m in mols]
    if only:
        pairs = [p for p in pairs if p.id in set(only)]
    return runner.pair_tasks(pairs, DATA10, labels=labels)


def run_kind(kind: str, argv: list[str]) -> int:
    args = cli.build_parser().parse_args([kind, *argv])
    prompt = load_prompt(args.prompt)
    mols = load_manifest()
    if kind == "eval":
        tasks = pair_task_list(mols, labels=(args.labels == "on"), only=args.only)
        run_kind_name, fmt = "pairs", args.format
    else:
        images = {m["id"]: m["images"] for m in mols}
        tasks = runner.image_tasks(images, DATA10, list(images), cli.PROBE_INSTRUCTION)
        args.format = fmt = "perception_json"
        run_kind_name = "perception"
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        cli._die("no tasks selected")
    settings = cli._settings(args)
    cli._check_budget(args, settings)
    if args.dry_run:
        cli._dry_run(tasks, prompt, settings, args.runs)
        return 0
    run_dir = cli.RUNS / args.name
    client = cli._client(settings, args)
    try:
        extra = {"data": SET, "labelled": False} | ({"labels": args.labels} if kind == "eval" else {})
        runner.init_run_dir(run_dir, cli._run_meta(args, run_kind_name, prompt, settings, **extra), prompt)
        summary = runner.run_tasks(tasks, run_dir=run_dir, prompt=prompt, fmt=fmt, client=client, runs=args.runs,
                                   workers=args.workers, max_calls=args.max_calls, retry_errors=not args.no_retry_errors,
                                   budget_usd=args.budget_usd)
    except (runner.RunMismatch, runner.BudgetExceeded) as exc:
        cli._die(str(exc))
    print(f"finished: {summary}")
    return 0


def by_task(run_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for rec in runner.load_records(run_dir / "outputs.jsonl").values():
        out.setdefault(rec["task_id"], []).append(rec)
    return {k: sorted(v, key=lambda r: r["run_idx"]) for k, v in out.items()}


def field_diff(field: str, ref: dict, ann: dict, label_norm=perception.normalise_label) -> str:
    """'ref only: ...; annotated only: ...' for one transcription field, empty when they agree.
    `label_norm` folds equivalent spellings of a label together (policy.canonical_label for the policy columns)."""
    if field in perception.INT_FIELDS:
        a, b = ref.get(field), ann.get(field)
        return "" if a == b else f"ref {a}, annotated {b}"
    norm = (lambda x: x) if field == perception.RING_FIELD else label_norm
    r = Counter(norm(x) for x in ref.get(field) or [])
    a = Counter(norm(x) for x in ann.get(field) or [])
    parts = []
    if r - a:
        parts.append("ref only " + ", ".join(map(str, sorted((r - a).elements(), key=str))))
    if a - r:
        parts.append("annotated only " + ", ".join(map(str, sorted((a - r).elements(), key=str))))
    return "; ".join(parts)


def cmd_summary(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="summary")
    ap.add_argument("--verify", required=True)
    ap.add_argument("--probe")
    ap.add_argument("--out", default=run_file("summary.md"))
    args = ap.parse_args(argv)
    verify = by_task(Path(args.verify))
    probe = by_task(Path(args.probe)) if args.probe else {}
    meta = json.loads((Path(args.verify) / "meta.json").read_text(encoding="utf-8-sig"))
    mols = load_manifest()
    L = ["# Google_Chem_10_tasks: verifier results", "",
         f"Prompt `{meta['prompt_name']}`, model `{meta['settings'].get('model')}`, thinking `{meta['settings'].get('thinking_level') or 'API default'}`, "
         f"{max((len(v) for v in verify.values()), default=0)} run(s) per pair. **No gold answers exist for these 10, so a verdict here is not scored right or wrong.**", "",
         "| Task | Verdict | Reason (run 0) | Cost | Time |", "|---|---|---|---|---|"]
    counts: Counter = Counter()
    spend = 0.0
    for m in mols:
        recs = verify.get(f"{m['id']}__annotated", [])
        if not recs:
            L.append(f"| {m['id']} | not run | | | |")
            counts["not run"] += 1
            continue
        shown = []
        for r in recs:
            spend += float((r.get("usage") or {}).get("cost") or 0)
            shown.append(str(r["decision"]) if r["status"] == "ok" and r.get("valid") else (r["status"] if r["status"] != "ok" else "invalid"))
            counts[shown[-1]] += 1
        first = recs[0]
        words = VERDICT_WORDS.get(first.get("decision"), shown[0]) if len(recs) == 1 else ""
        verdict = "/".join(shown) + (f" ({words})" if words else "")
        reason = (first.get("reason") or first.get("error") or "").replace("|", "/")
        cost = float((first.get("usage") or {}).get("cost") or 0)
        L.append(f"| {m['id']} | {verdict} | {reason} | ${cost:.3f} | {first.get('latency_s')}s |")
    L += ["", f"Verdict counts (calls): {dict(counts)}. Total spend on verdicts: ${spend:.3f}.", ""]
    if probe:
        L += ["## Perception cross-check (reference vs annotated, transcribed independently)", "",
              "Each image was transcribed on its own by the same model. Differences in the critical fields are places "
              "where the two drawings differ or where the model misread one of them; both explanations need a look at the images. "
              "This is a cross-check on the verdict, not gold.", "",
              "| Task | Annotated ink | Annotated legibility | Critical fields agree | Differences |", "|---|---|---|---|---|"]
        for m in mols:
            ref = next((r for r in probe.get(f"{m['id']}/ref", []) if r["status"] == "ok" and r.get("data")), None)
            ann = next((r for r in probe.get(f"{m['id']}/annotated", []) if r["status"] == "ok" and r.get("data")), None)
            if not ref or not ann:
                L.append(f"| {m['id']} | | | probe missing | |")
                continue
            rd, ad = ref["data"], ann["data"]
            diffs = [f"**{f}**: {d}" for f in perception.CRITICAL_FIELDS if (d := field_diff(f, rd, ad))]
            agree = "yes" if not diffs else f"{len(perception.CRITICAL_FIELDS) - len(diffs)}/{len(perception.CRITICAL_FIELDS)}"
            L.append(f"| {m['id']} | {ad.get('ink_colour')} | {ad.get('legibility')} | {agree} | {'; '.join(diffs)} |")
        L.append("")
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"written: {args.out}")
    return 0


CONFIDENCE_RE = re.compile(r"(?im)^\W*confidence\W*[:=]?\s*(\d{1,3})\b")
CSV_COLUMNS = ["IDs", "Samples", "Annotated", "verdict", "verdict_label", "reason", "run_decisions", "n_runs", "accept_votes",
               "reject_votes", "cannot_verify_votes", "agreement", "stated_confidence", "conf_prompt_verdict", "model", "thinking_level",
               "ref_aromatic_depiction", "annotated_aromatic_depiction", "annotated_ink", "annotated_legibility",
               "critical_fields_agree", "perception_differences", "cost_usd", "latency_s"]


def legacy_rows(mols: list[dict], verify: dict, probe: dict, conf: dict, meta: dict) -> list[dict]:
    """The results-CSV rows as they were before the decision layer existed (columns CSV_COLUMNS).
    Keys that start with '_' carry data for later steps and are never written to a file."""
    rows = []
    for m in mols:
        recs = verify.get(f"{m['id']}__annotated", [])
        first = recs[0] if recs else {}
        decisions = [str(r["decision"]) if r["status"] == "ok" and r.get("valid") else (r["status"] if r["status"] != "ok" else "invalid") for r in recs]
        row = {"IDs": m["id"], "Samples": m["images"]["ref"]["source"], "Annotated": m["images"]["annotated"]["source"],
               "verdict": decisions[0] if decisions else "not run",
               "verdict_label": VERDICT_WORDS.get(first.get("decision"), decisions[0] if decisions else "not run"),
               "reason": first.get("reason") or first.get("error") or "", "run_decisions": "/".join(decisions),
               "n_runs": len(decisions), "accept_votes": decisions.count("1"), "reject_votes": decisions.count("0"),
               "cannot_verify_votes": decisions.count("2"),
               "agreement": round(max(Counter(decisions).values()) / len(decisions), 2) if decisions else "",
               "model": meta["settings"].get("model"), "thinking_level": meta["settings"].get("thinking_level") or "API default",
               "cost_usd": round(sum(float((r.get("usage") or {}).get("cost") or 0) for r in recs), 4),
               "latency_s": first.get("latency_s", "")}
        crec = next((r for r in conf.get(f"{m['id']}__annotated", []) if r["status"] == "ok" and r.get("valid")), None)
        if crec:
            found = CONFIDENCE_RE.search(crec.get("raw_text") or "")
            value = int(found.group(1)) if found else None
            row["stated_confidence"] = value if value is not None and value <= 100 else ""
            row["conf_prompt_verdict"] = crec["decision"]
        ref = next((r for r in probe.get(f"{m['id']}/ref", []) if r["status"] == "ok" and r.get("data")), None)
        ann = next((r for r in probe.get(f"{m['id']}/annotated", []) if r["status"] == "ok" and r.get("data")), None)
        if ref and ann:
            rd, ad = ref["data"], ann["data"]
            diffs = [f"{f}: {d}" for f in perception.CRITICAL_FIELDS if (d := field_diff(f, rd, ad))]
            row.update(ref_aromatic_depiction=rd.get("aromatic_ring_depiction"), annotated_aromatic_depiction=ad.get("aromatic_ring_depiction"),
                       annotated_ink=ad.get("ink_colour"), annotated_legibility=ad.get("legibility"),
                       critical_fields_agree="yes" if not diffs else f"{len(perception.CRITICAL_FIELDS) - len(diffs)}/{len(perception.CRITICAL_FIELDS)}",
                       perception_differences="; ".join(diffs), _probe=(rd, ad))
        rows.append(row)
    return rows


def write_csv(path: str, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:  # BOM so Excel reads the UTF-8 correctly
        writer = csv.DictWriter(fh, fieldnames=columns, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cmd_csv(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="csv")
    ap.add_argument("--verify", required=True)
    ap.add_argument("--probe")
    ap.add_argument("--verify-conf", help="run made with prompts/baseline_v0_conf.txt (adds the model's stated confidence)")
    ap.add_argument("--out", default=run_file("results.csv"))
    args = ap.parse_args(argv)
    meta = json.loads((Path(args.verify) / "meta.json").read_text(encoding="utf-8-sig"))
    rows = legacy_rows(load_manifest(), by_task(Path(args.verify)), by_task(Path(args.probe)) if args.probe else {},
                       by_task(Path(args.verify_conf)) if args.verify_conf else {}, meta)
    write_csv(args.out, CSV_COLUMNS, rows)
    print(f"written: {args.out} ({len(rows)} rows)")
    return 0


POLICY_PROMPT = ROOT / "prompts" / "chem_equiv_policy_v1.txt"
DECISION_COLUMNS = ["final_outcome", "decision_rule", "requires_review", "failure_categories", "rejection_evidence",
                    "notation_differences", "flags", "vote_agreement", "votes", "model_confidence", "structured_calls",
                    "needs_escalation", "graph_check", "policy_version", "perception_differences_policy"]


# A compact view for reading: the decision first, the old verdict only for comparison. Every name is also in
# CSV_COLUMNS + DECISION_COLUMNS, so the full file keeps its original layout for anything that reads it by position.
FINAL_COLUMNS = ["IDs", "final_outcome", "requires_review", "decision_rule", "failure_categories", "rejection_evidence",
                 "notation_differences", "votes", "vote_agreement", "model_confidence", "structured_calls", "verdict_label",
                 "Samples", "Annotated"]


def confusion(pairs: list[tuple[bool, bool]]) -> dict:
    """(accepted, actually correct) pairs -> counts and rates. 'Positive' means the drawing is correct and should be
    accepted, so a false accept (FP) is a wrong drawing that got through. A rate is None when its denominator is 0."""
    tp = sum(1 for a, c in pairs if a and c)
    fp = sum(1 for a, c in pairs if a and not c)
    fn = sum(1 for a, c in pairs if not a and c)
    tn = sum(1 for a, c in pairs if not a and not c)

    def ratio(num, den):
        return round(num / den, 3) if den else None
    return {"n": len(pairs), "TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "specificity": ratio(tn, tn + fp), "accuracy": ratio(tp + tn, len(pairs))}


def cmd_metrics(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="metrics")
    ap.add_argument("--results", default=run_file("results.csv"))
    ap.add_argument("--labels", default=str(DATA10 / "provisional_labels.csv"), help="CSV with IDs and correct = yes / no / unknown")
    ap.add_argument("--out", default=run_file("metrics.csv"))
    args = ap.parse_args(argv)
    with open(args.labels, newline="", encoding="utf-8-sig") as fh:
        labels = {r["IDs"].strip(): r["correct"].strip().lower() for r in csv.DictReader(fh)}
    with open(args.results, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    known = [r for r in rows if labels.get(r["IDs"]) in ("yes", "no")]
    skipped = [r["IDs"] for r in rows if labels.get(r["IDs"]) not in ("yes", "no")]
    systems = {"old prompt, first run (verdict)": lambda r: r["verdict"] == "1",
               "decision layer (final_outcome)": lambda r: r.get("final_outcome") == "ACCEPT"}
    out = []
    for name, accepted in systems.items():
        result = confusion([(accepted(r), labels[r["IDs"]] == "yes") for r in known])
        out.append({"system": name, **result, "left_out": "; ".join(skipped)})
        print(f"\n{name}   (n={result['n']}, left out: {', '.join(skipped) or 'none'})")
        print(f"{'':32s}{'actually correct':>18s}{'actually wrong':>16s}")
        print(f"{'accepted':32s}{'TP = ' + str(result['TP']):>18s}{'FP = ' + str(result['FP']):>16s}")
        print(f"{'not accepted (reject/uncertain)':32s}{'FN = ' + str(result['FN']):>18s}{'TN = ' + str(result['TN']):>16s}")
        print("precision {precision}  recall {recall}  F1 {f1}  specificity {specificity}  accuracy {accuracy}".format(**result))
    write_csv(args.out, ["system", "n", "TP", "FP", "FN", "TN", "precision", "recall", "f1", "specificity", "accuracy", "left_out"], out)
    print(f"\nwritten: {args.out}")
    return 0


def cmd_policy(argv: list[str]) -> int:
    """Write (or check) the policy prompt file. The file is generated from chemverify.policy, never edited by hand."""
    ap = argparse.ArgumentParser(prog="policy")
    ap.add_argument("--write", action="store_true", help=f"write {POLICY_PROMPT.name} from the policy")
    args = ap.parse_args(argv)
    text = policy.system_prompt()
    if args.write:
        with open(POLICY_PROMPT, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    in_sync = POLICY_PROMPT.exists() and POLICY_PROMPT.read_text(encoding="utf-8-sig") == text
    print(f"{POLICY_PROMPT.name}: {'in sync with chemverify.policy' if in_sync else 'OUT OF SYNC (run: policy --write)'} (policy {policy.POLICY_VERSION})")
    return 0 if in_sync else 1


def judgements_for(mol_id: str, verify: dict, policy_recs: dict) -> list:
    """Every recorded call for one task as a Judgement. Transient API errors are skipped: they are retried, not counted."""
    task = f"{mol_id}__annotated"
    return ([decision.judge_legacy(r) for r in verify.get(task, []) if r.get("status") != "error"]
            + [decision.judge_structured(r) for r in policy_recs.get(task, []) if r.get("status") != "error"])


def compute_decisions(mols: list[dict], verify: dict, policy_recs: dict, smiles: dict) -> dict:
    return {m["id"]: decision.decide(judgements_for(m["id"], verify, policy_recs), graphcheck.compare_smiles(*smiles.get(m["id"], ("", ""))))
            for m in mols}


def decision_columns(d: decision.Decision, legacy_row: dict) -> dict:
    cols = {"final_outcome": d.outcome, "decision_rule": d.rule, "requires_review": "yes" if d.requires_review else "no",
            "failure_categories": "; ".join(d.failure_categories), "rejection_evidence": decision.evidence_json(d.evidence),
            "notation_differences": "; ".join(d.notation_differences), "flags": "; ".join(d.flags),
            "vote_agreement": "" if d.vote_agreement is None else d.vote_agreement, "votes": decision.votes_text(d.votes),
            "model_confidence": "" if d.model_confidence is None else d.model_confidence, "structured_calls": d.structured_calls,
            "needs_escalation": "yes" if d.needs_escalation else "no", "graph_check": d.graph, "policy_version": policy.POLICY_VERSION,
            "perception_differences_policy": ""}
    if legacy_row.get("_probe"):
        rd, ad = legacy_row["_probe"]
        fields = [f for f in perception.CRITICAL_FIELDS if f not in policy.NOTATION_ONLY_FIELDS]
        cols["perception_differences_policy"] = "; ".join(f"{f}: {x}" for f in fields if (x := field_diff(f, rd, ad, policy.canonical_label)))
    return cols


def escalate(args, mols: list[dict], verify: dict, policy_dir: Path, smiles: dict) -> None:
    """Spend model calls only on tasks whose decision is still unresolved, one structured call per task per round."""
    if not (POLICY_PROMPT.exists() and POLICY_PROMPT.read_text(encoding="utf-8-sig") == policy.system_prompt()):
        cli._die(f"{POLICY_PROMPT.name} is missing or does not match chemverify.policy; run `pipeline_google10.py policy --write`")
    prompt = load_prompt(str(POLICY_PROMPT))
    ns = cli.build_parser().parse_args(["eval", "--prompt", str(POLICY_PROMPT), "--format", "verdict_json", "--name", policy_dir.name,
                                        "--runs", "1", "--workers", str(args.workers), "--thinking-level", args.thinking_level])
    ns.budget_usd = args.budget_usd
    settings = cli._settings(ns)
    cli._check_budget(ns, settings)
    client = None
    spent = 0.0
    for _ in range(policy.MAX_STRUCTURED_CALLS):
        policy_recs = by_task(policy_dir) if (policy_dir / "outputs.jsonl").exists() else {}
        decisions = compute_decisions(mols, verify, policy_recs, smiles)
        todo = [m for m in mols if decisions[m["id"]].needs_escalation and (not args.only or m["id"] in args.only)]
        if not todo:
            print("escalation: nothing unresolved" if spent == 0 else "escalation: every task is resolved or out of calls")
            return
        groups: dict[int, list[dict]] = {}
        for m in todo:
            groups.setdefault(decisions[m["id"]].structured_calls, []).append(m)
        for done_calls, group in sorted(groups.items()):
            tasks = pair_task_list(group, labels=True)
            print(f"escalation: {len(tasks)} unresolved task(s) get structured call #{done_calls + 1}: {', '.join(m['id'] for m in group)}", file=sys.stderr)
            if args.dry_run:
                cli._dry_run(tasks, prompt, settings, done_calls + 1)
                continue
            if client is None:
                client = cli._client(settings, ns)
                runner.init_run_dir(policy_dir, cli._run_meta(ns, "pairs", prompt, settings, data=SET, labelled=False, labels="on",
                                                              policy=policy.POLICY_VERSION), prompt)
            remaining = None if args.budget_usd is None else max(args.budget_usd - spent, 0.0001)
            summary = runner.run_tasks(tasks, run_dir=policy_dir, prompt=prompt, fmt="verdict_json", client=client, runs=done_calls + 1,
                                       workers=args.workers, max_calls=args.max_calls, retry_errors=True, budget_usd=remaining)
            spent += summary.get("spent_usd", 0.0)
            print(f"escalation: {summary}", file=sys.stderr)
            if summary.get("stopped_by_budget"):
                print(f"escalation stopped: budget of ${args.budget_usd:.2f} reached; unresolved tasks stay UNCERTAIN", file=sys.stderr)
                return
        if args.dry_run:
            return


def cmd_decide(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="decide")
    ap.add_argument("--verify", required=True, help="existing digit-plus-reason run (reused, not repeated)")
    ap.add_argument("--probe")
    ap.add_argument("--verify-conf")
    ap.add_argument("--policy-run", default=run_file("policy"), help="run folder for structured policy calls")
    ap.add_argument("--smiles-csv", help="optional CSV (IDs, Samples_SMILES, Annotated_SMILES) enabling the RDKit graph check")
    ap.add_argument("--out", default=run_file("results.csv"))
    ap.add_argument("--json-out", default=run_file("decisions.json"))
    ap.add_argument("--final-out", default=run_file("final.csv"), help="compact CSV with the decision columns first")
    ap.add_argument("--escalate", action="store_true", help="make structured model calls for unresolved tasks (costs money; default: none)")
    ap.add_argument("--only", nargs="+", help="escalate only these molecule ids")
    ap.add_argument("--thinking-level", default="low")
    ap.add_argument("--budget-usd", type=float, default=None, help="stop starting new calls after this much spend (default: no cap)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-calls", type=int, default=1000, help="refuse to start more calls than this in one round (accident guard)")
    ap.add_argument("--dry-run", action="store_true", help="with --escalate: show the calls that would be made, send nothing")
    args = ap.parse_args(argv)
    mols = load_manifest()
    verify = by_task(Path(args.verify))
    probe = by_task(Path(args.probe)) if args.probe else {}
    conf = by_task(Path(args.verify_conf)) if args.verify_conf else {}
    meta = json.loads((Path(args.verify) / "meta.json").read_text(encoding="utf-8-sig"))
    smiles = graphcheck.load_smiles(args.smiles_csv) if args.smiles_csv else {}
    policy_dir = Path(args.policy_run)
    if args.escalate:
        escalate(args, mols, verify, policy_dir, smiles)
    policy_recs = by_task(policy_dir) if (policy_dir / "outputs.jsonl").exists() else {}
    decisions = compute_decisions(mols, verify, policy_recs, smiles)
    rows = legacy_rows(mols, verify, probe, conf, meta)
    for row, m in zip(rows, mols):
        row.update(decision_columns(decisions[m["id"]], row))
    write_csv(args.out, CSV_COLUMNS + DECISION_COLUMNS, rows)
    write_csv(args.final_out, FINAL_COLUMNS, rows)
    Path(args.json_out).write_text(json.dumps({"policy_version": policy.POLICY_VERSION, "decisions": {k: v.to_dict() for k, v in decisions.items()}},
                                              indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{'task':13s} {'outcome':10s} {'rule':24s} {'structured':>10s} {'escalate':>8s}  failure categories / causes")
    for m in mols:
        d = decisions[m["id"]]
        print(f"{m['id']:13s} {d.outcome:10s} {d.rule:24s} {d.structured_calls:>10d} {'yes' if d.needs_escalation else 'no':>8s}  {'; '.join(d.failure_categories)}")
    print(f"written: {args.out} ({len(rows)} rows, {len(CSV_COLUMNS)} original + {len(DECISION_COLUMNS)} decision columns); {args.json_out}")
    print(f"written: {args.final_out} (compact, decision columns first)")
    return 0


def main(argv: list[str]) -> int:
    commands = {"prep": cmd_prep, "summary": cmd_summary, "csv": cmd_csv, "policy": cmd_policy, "decide": cmd_decide, "metrics": cmd_metrics,
                "verify": lambda a: run_kind("eval", a), "probe": lambda a: run_kind("probe", a)}
    argv = list(argv)
    if "--set" in argv:  # e.g. `--set google30 prep --csv ...`: data/<set>/, runs/<set>_*, outputs named after the set
        position = argv.index("--set")
        if position + 1 >= len(argv):
            cli._die("--set needs a dataset name")
        use_dataset(argv[position + 1])
        del argv[position:position + 2]
    if not argv or argv[0] not in commands:
        print(__doc__)
        return 2
    return commands[argv[0]](argv[1:])


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main(sys.argv[1:]))
