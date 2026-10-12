"""Verifier pipeline for the Google-Chem CSVs (columns IDs, Samples, Annotated).

Samples is the typeset reference (Image A), Annotated is the annotator's drawing (Image B). Nobody has labelled which
drawings are right, so this produces decisions with their evidence, not accuracy numbers. The old digit-plus-reason
prompt and the perception probe are no longer part of this pipeline; the code that did that is kept, unchanged, in
archive/pipeline_google10_v1_before_confirmation.py. Authoritative policy: src/chemverify/policy.py (the prompt files in
prompts/ are generated from it; drafts that nothing reads are in prompts/archive/).

    python -I pipeline_google10.py --set google30 prep --csv <Google_Chem_30_tasks.csv>
    python -I pipeline_google10.py policy [--write]            # generate / check prompts/chem_equiv_{policy,confirm}_v1.txt
    python -I pipeline_google10.py --set google30 --run-tag v2 decide [--escalate ...]
    python -I pipeline_google10.py --set google30 --run-tag v2 metrics --labels <csv>      # optional, needs labels

`--set NAME` keeps one CSV's images in data/NAME/. `--run-tag TAG` puts every output and run folder under runs/NAME_TAG_*
so an earlier run is never touched; `decide` refuses to write over an existing file (or into a recorded run) unless told to.

decide
    no flag            recomputes decisions from recorded calls; makes no model call.
    --evidence-from D  read the policy-prompt calls from recorded run folder D (read-only). No new policy calls are made.
    --escalate         make live calls: policy-prompt calls for unresolved pairs (unless --evidence-from) and one
                       confirming call for every pair whose only objection is an unconfirmed reject. Confirming calls
                       are refused until you pass --approve-confirm-prompt <first 12 characters of the prompt's sha>.
    --no-confirm       never make confirming calls; such pairs end as UNCERTAIN confirmation_not_run.
    --dry-run          with --escalate: show the calls that would be made, send nothing, write nothing.
    --budget-usd N     stop starting new calls after N dollars (default: no cap).
Every run prints, and writes to the decisions JSON, the number of calls and the cost of each stage.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from chemverify import cli, decision, graphcheck, policy, runner  # noqa: E402
from chemverify.download import DownloadError, atomic_write, urllib_fetch  # noqa: E402
from chemverify.gemini import TextPart  # noqa: E402
from chemverify.imageinfo import sniff  # noqa: E402
from chemverify.manifest import ALLOWED_HOST_SUFFIXES, ID_RE, check_host, download_url, sanitize_url  # noqa: E402
from chemverify.pairs import Pair  # noqa: E402
from chemverify.prompts import load_prompt  # noqa: E402

SET = "google10"  # dataset name; `--set google30` (first argument) keeps another CSV's images apart
RUN_TAG = ""  # `--run-tag v2` puts outputs under runs/<set>_v2_*
DATA10 = ROOT / "data" / SET
MANIFEST10 = DATA10 / "manifest.json"
ROLES10 = ("ref", "annotated")
COLUMNS_IN = {"ids": "id", "samples": "ref", "annotated": "annotated"}
POLICY_PROMPT = ROOT / "prompts" / "chem_equiv_policy_v1.txt"
CONFIRM_PROMPT = ROOT / "prompts" / "chem_equiv_confirm_v1.txt"
PROMPT_FILES = {POLICY_PROMPT: policy.system_prompt, CONFIRM_PROMPT: policy.confirmation_prompt}


def use_dataset(name: str) -> None:
    global SET, DATA10, MANIFEST10
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,30}", name):
        cli._die(f"--set must be lower-case letters, digits and underscores, got {name!r}")
    SET, DATA10 = name, ROOT / "data" / name
    MANIFEST10 = DATA10 / "manifest.json"


def use_tag(tag: str) -> None:
    global RUN_TAG
    if tag and not re.fullmatch(r"[a-z][a-z0-9_]{0,20}", tag):
        cli._die(f"--run-tag must be lower-case letters, digits and underscores, got {tag!r}")
    RUN_TAG = tag


def run_file(suffix: str) -> str:
    """runs/<set>[_<tag>]_<suffix>, the default location of every output of this dataset and tag."""
    prefix = f"{SET}_{RUN_TAG}" if RUN_TAG else SET
    return str(ROOT / "runs" / f"{prefix}_{suffix}")


# --------------------------------------------------------------------------- input and images


def read_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        mapping = {n: COLUMNS_IN[n.strip().lower()] for n in (reader.fieldnames or []) if n and n.strip().lower() in COLUMNS_IN}
        missing = sorted(set(COLUMNS_IN.values()) - set(mapping.values()))
        if missing:
            cli._die(f"CSV is missing columns for {missing}; expected headers {list(COLUMNS_IN)}")
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


def confirm_task(mol: dict, claim: dict) -> runner.Task:
    """The confirming call for one reported difference: the same two images plus the claim as a hypothesis to test."""
    base = pair_task_list([mol], labels=True)[0]
    key = decision.claim_key(claim["category"], claim["location"])
    message = policy.claim_message(claim)
    fingerprint = hashlib.sha256(f"{base.input_fp}|{key}|{message}".encode()).hexdigest()[:12]
    meta = {**base.meta, "kind": "confirm", "claim_key": key, "claim_category": claim["category"], "claim_location": claim["location"],
            "claim_reference": claim.get("reference", ""), "claim_annotated": claim.get("annotated", "")}
    return runner.Task(f"{mol['id']}__confirm", meta, lambda: base.load_parts() + [TextPart(message)], fingerprint)


def by_task(run_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for rec in runner.load_records(Path(run_dir) / "outputs.jsonl").values():
        out.setdefault(rec["task_id"], []).append(rec)
    return {k: sorted(v, key=lambda r: r["run_idx"]) for k, v in out.items()}


# --------------------------------------------------------------------------- prompts and guards


def prompt_status() -> dict[str, bool]:
    """Is each generated prompt file exactly what chemverify.policy renders?"""
    return {path.name: path.exists() and path.read_text(encoding="utf-8-sig") == render() for path, render in PROMPT_FILES.items()}


def prompt_sha(path: Path) -> str:
    return load_prompt(str(path)).sha[:12] if path.exists() else "-"


def cmd_policy(argv: list[str]) -> int:
    """Write (or check) the generated prompt files. They are produced from chemverify.policy and never edited by hand."""
    ap = argparse.ArgumentParser(prog="policy")
    ap.add_argument("--write", action="store_true", help="write both prompt files from the policy")
    args = ap.parse_args(argv)
    if args.write:
        for path, render in PROMPT_FILES.items():
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(render())
    status = prompt_status()
    for path in PROMPT_FILES:
        print(f"{path.name}: {'in sync with chemverify.policy' if status[path.name] else 'MISSING or OUT OF SYNC (run: policy --write)'} "
              f"(sha {prompt_sha(path)})")
    print(f"policy {policy.POLICY_VERSION} (prompt text), decision rules {policy.DECISION_VERSION}, csv schema {policy.SCHEMA_VERSION}")
    return 0 if all(status.values()) else 1


def new_output(path: str, overwrite: bool) -> str:
    """An output path that does not exist yet (so an earlier run's files are never replaced by accident)."""
    if Path(path).exists() and not overwrite:
        cli._die(f"{path} already exists. Use a different --run-tag (or --overwrite if you really mean to replace it).")
    return path


def writable_run_dir(path: Path, resume: bool) -> None:
    """New calls go only into run folders made by this pipeline version; a recorded run is never appended to."""
    if not path.exists():
        return
    if not resume:
        cli._die(f"{path} already exists. Pass --resume to continue a run made by this pipeline, or choose another --run-tag.")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8-sig")) if (path / "meta.json").exists() else {}
    if not meta.get("decision_version"):
        cli._die(f"{path} was not created by this pipeline version (a recorded run): refusing to write into it.")


# --------------------------------------------------------------------------- cost tracking


def call_cost(rec: dict) -> float:
    return float((rec.get("usage") or {}).get("cost") or 0)


def cost_ledger(mols: list[dict], policy_recs: dict, confirm_recs: dict) -> dict:
    """Per pair and per stage: how many calls were recorded and what they cost (the cost the API reported)."""
    per = {}
    for m in mols:
        p = [r for r in policy_recs.get(f"{m['id']}__annotated", []) if r.get("status") != "error"]
        c = [r for r in confirm_recs.get(f"{m['id']}__confirm", []) if r.get("status") != "error"]
        per[m["id"]] = {"policy_calls": len(p), "policy_cost_usd": round(sum(map(call_cost, p)), 4),
                        "confirm_calls": len(c), "confirm_cost_usd": round(sum(map(call_cost, c)), 4)}
    totals = {k: round(sum(v[k] for v in per.values()), 4) for k in ("policy_calls", "policy_cost_usd", "confirm_calls", "confirm_cost_usd")}
    totals["total_cost_usd"] = round(totals["policy_cost_usd"] + totals["confirm_cost_usd"], 4)
    return {"per_pair": per, "totals": totals}


# --------------------------------------------------------------------------- decisions


def judgements_for(mol_id: str, policy_recs: dict) -> list:
    """Every recorded policy-prompt call for one pair. Transient API errors are skipped: they are retried, not counted."""
    return [decision.judge_structured(r) for r in policy_recs.get(f"{mol_id}__annotated", []) if r.get("status") != "error"]


def confirmation_for(mol_id: str, confirm_recs: dict, claim: dict | None):
    """The latest confirming call that tested exactly this claim, or None."""
    if not claim:
        return None
    key = decision.claim_key(claim["category"], claim["location"])
    tested = [r for r in confirm_recs.get(f"{mol_id}__confirm", []) if r.get("status") != "error" and r.get("claim_key") == key]
    return decision.judge_confirmation(tested[-1]) if tested else None


def compute_decisions(mols: list[dict], policy_recs: dict, confirm_recs: dict, smiles: dict, *, final: bool) -> dict:
    """final=False: the working decision that drives escalation. final=True: what may be exported (explanation and gate applied)."""
    out = {}
    for m in mols:
        js = judgements_for(m["id"], policy_recs)
        graph = graphcheck.compare_smiles(*smiles.get(m["id"], ("", "")))
        claim = decision.decide(js, None, graph, confirmation_enabled=False).claim
        conf = confirmation_for(m["id"], confirm_recs, claim)
        out[m["id"]] = decision.publish(js, conf, graph, final=True) if final else decision.decide(js, conf, graph)
    return out


COLUMNS = ["IDs", "Samples", "Annotated", "final_outcome", "requires_review", "decision_rule", "evidence_status", "consistency_status",
           "consistency_notes", "failure_categories", "rejection_evidence", "final_explanation", "notation_differences", "votes",
           "structured_calls", "confirm_calls", "accept_share", "model_confidence", "flags", "policy_cost_usd", "confirm_cost_usd",
           "api_cost_usd", "evidence_source", "graph_check", "policy_version", "decision_version", "schema_version"]
# The compact reading view: the decision, its evidence and its explanation first.
FINAL_COLUMNS = ["IDs", "final_outcome", "requires_review", "decision_rule", "evidence_status", "failure_categories", "final_explanation",
                 "rejection_evidence", "notation_differences", "votes", "confirm_calls", "api_cost_usd", "evidence_source", "Samples", "Annotated"]


def build_rows(mols: list[dict], decisions: dict, ledger: dict, evidence_source: str) -> list[dict]:
    rows = []
    for m in mols:
        d, c = decisions[m["id"]], ledger["per_pair"][m["id"]]
        rows.append({
            "IDs": m["id"], "Samples": m["images"]["ref"]["source"], "Annotated": m["images"]["annotated"]["source"],
            "final_outcome": d.outcome, "requires_review": "yes" if d.requires_review else "no", "decision_rule": d.rule,
            "evidence_status": d.evidence_status, "consistency_status": d.consistency_status, "consistency_notes": "; ".join(d.consistency),
            "failure_categories": "; ".join(d.failure_categories), "rejection_evidence": decision.evidence_json(d.evidence),
            "final_explanation": d.explanation, "notation_differences": "; ".join(d.notation_differences), "votes": decision.votes_text(d.votes),
            "structured_calls": d.structured_calls, "confirm_calls": c["confirm_calls"],
            "accept_share": "" if d.accept_share is None else d.accept_share,
            "model_confidence": "" if d.model_confidence is None else d.model_confidence, "flags": "; ".join(d.flags),
            "policy_cost_usd": c["policy_cost_usd"], "confirm_cost_usd": c["confirm_cost_usd"],
            "api_cost_usd": round(c["policy_cost_usd"] + c["confirm_cost_usd"], 4), "evidence_source": evidence_source,
            "graph_check": d.graph, "policy_version": policy.POLICY_VERSION, "decision_version": policy.DECISION_VERSION,
            "schema_version": policy.SCHEMA_VERSION})
    return rows


def write_csv(path: str, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:  # BOM so Excel reads the UTF-8 correctly
        writer = csv.DictWriter(fh, fieldnames=columns, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- live calls (only with --escalate)


def stage_context(args, prompt_path: Path, fmt: str, name: str):
    """Prompt, parsed-args namespace and settings for one stage (policy calls or confirming calls)."""
    prompt = load_prompt(str(prompt_path))
    ns = cli.build_parser().parse_args(["eval", "--prompt", str(prompt_path), "--format", "verdict_json", "--name", name, "--runs", "1",
                                        "--workers", str(args.workers), "--thinking-level", args.thinking_level, "--backend", args.backend])
    ns.format = fmt
    ns.budget_usd = args.budget_usd
    settings = cli._settings(ns)
    cli._check_budget(ns, settings)
    return prompt, ns, settings


def confirmations_allowed(args) -> bool:
    """Confirming calls need the exact prompt hash the user approved; without it none are made."""
    if args.no_confirm:
        return False
    actual = load_prompt(str(CONFIRM_PROMPT)).sha[:12]
    if not args.approve_confirm_prompt:
        cli._err(f"confirming calls are NOT made: pass --approve-confirm-prompt {actual} once you have read prompts/{CONFIRM_PROMPT.name}")
        return False
    if args.approve_confirm_prompt != actual:
        cli._die(f"--approve-confirm-prompt {args.approve_confirm_prompt} does not match prompts/{CONFIRM_PROMPT.name} (sha {actual})")
    return True


def escalate(args, mols: list[dict], policy_dir: Path, confirm_dir: Path, smiles: dict, evidence_recs, *, client=None) -> dict:
    """Spend model calls only where a decision is still open: policy-prompt calls for unresolved pairs (never when the
    evidence is a recorded run) and one confirming call for each unconfirmed reject. Returns calls and cost made here."""
    reuse = evidence_recs is not None
    made = {"policy_calls": 0, "policy_usd": 0.0, "confirm_calls": 0, "confirm_usd": 0.0}
    spent_total = 0.0
    confirm_ok = confirmations_allowed(args) or (args.dry_run and not args.no_confirm)  # a dry run only previews, so it may list them
    ctx: dict = {}

    def run(stage: str, tasks: list, runs: int, prompt_path: Path, fmt: str, name: str, run_dir: Path) -> None:
        nonlocal client, spent_total
        if stage not in ctx:
            ctx[stage] = stage_context(args, prompt_path, fmt, name)
        prompt, ns, settings = ctx[stage]
        if args.dry_run:
            cli._dry_run(tasks, prompt, settings, runs)
            return
        if client is None:
            client = cli._client(settings, ns)
        if not (run_dir / "meta.json").exists():
            runner.init_run_dir(run_dir, cli._run_meta(ns, "pairs", prompt, settings, data=SET, labelled=False, labels="on", stage=stage,
                                                       policy=policy.POLICY_VERSION, decision_version=policy.DECISION_VERSION,
                                                       schema_version=policy.SCHEMA_VERSION,
                                                       approved_prompt_sha=args.approve_confirm_prompt if stage == "confirm" else None), prompt)
        remaining = None if args.budget_usd is None else max(args.budget_usd - spent_total, 0.0001)
        summary = runner.run_tasks(tasks, run_dir=run_dir, prompt=prompt, fmt=fmt, client=client, runs=runs, workers=args.workers,
                                   max_calls=args.max_calls, retry_errors=True, budget_usd=remaining)
        spent = summary.get("spent_usd", 0.0)
        spent_total += spent
        made[f"{'policy' if stage == 'policy' else 'confirm'}_usd"] += spent
        made[f"{'policy' if stage == 'policy' else 'confirm'}_calls"] += len(tasks)
        print(f"{stage}: {summary}", file=sys.stderr)
        if summary.get("stopped_by_budget"):
            print(f"stopped: budget of ${args.budget_usd:.2f} reached; unresolved pairs stay UNCERTAIN", file=sys.stderr)
            raise StopIteration

    try:
        for _ in range(policy.MAX_STRUCTURED_CALLS + policy.MAX_CONFIRMATION_CALLS):
            policy_recs = evidence_recs if reuse else (by_task(policy_dir) if (policy_dir / "outputs.jsonl").exists() else {})
            confirm_recs = by_task(confirm_dir) if (confirm_dir / "outputs.jsonl").exists() else {}
            decisions = compute_decisions(mols, policy_recs, confirm_recs, smiles, final=False)
            wanted = (lambda m: not args.only or m["id"] in args.only)
            policy_todo = [] if reuse else [m for m in mols if decisions[m["id"]].needs_escalation and wanted(m)]
            confirm_todo = [m for m in mols if confirm_ok and decisions[m["id"]].needs_confirmation and wanted(m)
                            and len(confirm_recs.get(f"{m['id']}__confirm", [])) < policy.MAX_CONFIRMATION_CALLS]
            if not policy_todo and not confirm_todo:
                break
            groups: dict[int, list[dict]] = {}
            for m in policy_todo:
                groups.setdefault(decisions[m["id"]].structured_calls, []).append(m)
            for done_calls, group in sorted(groups.items()):
                tasks = pair_task_list(group, labels=True)
                print(f"policy call #{done_calls + 1} for {len(tasks)} unresolved pair(s): {', '.join(m['id'] for m in group)}", file=sys.stderr)
                run("policy", tasks, done_calls + 1, POLICY_PROMPT, "verdict_json", policy_dir.name, policy_dir)
            if confirm_todo:
                tasks = [confirm_task(m, decisions[m["id"]].claim) for m in confirm_todo]
                print(f"confirming call for {len(tasks)} unconfirmed reject(s): {', '.join(m['id'] for m in confirm_todo)}", file=sys.stderr)
                run("confirm", tasks, 1, CONFIRM_PROMPT, "claim_json", confirm_dir.name, confirm_dir)
            if args.dry_run:
                break
    except StopIteration:
        pass
    return made


def cmd_decide(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="decide")
    ap.add_argument("--evidence-from", help="recorded run folder with policy-prompt calls, read-only (no new policy calls are made)")
    ap.add_argument("--policy-run", default=None, help="run folder for new policy-prompt calls (default: runs/<set>[_<tag>]_policy)")
    ap.add_argument("--confirm-run", default=None, help="run folder for confirming calls (default: runs/<set>[_<tag>]_confirm)")
    ap.add_argument("--smiles-csv", help="optional CSV (IDs, Samples_SMILES, Annotated_SMILES) enabling the RDKit graph check")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--final-out", default=None, help="compact CSV with the decision, evidence and explanation first")
    ap.add_argument("--escalate", action="store_true", help="make live model calls (costs money; default: none)")
    ap.add_argument("--no-confirm", action="store_true", help="never make confirming calls")
    ap.add_argument("--approve-confirm-prompt", metavar="SHA12", help="first 12 characters of the confirming prompt's sha, once you approved its text")
    ap.add_argument("--only", nargs="+", help="make live calls only for these molecule ids")
    ap.add_argument("--thinking-level", default="low")
    ap.add_argument("--backend", default="auto", choices=("auto", "gemini", "openrouter"))
    ap.add_argument("--budget-usd", type=float, default=None, help="stop starting new calls after this much spend (default: no cap)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-calls", type=int, default=1000, help="refuse to start more calls than this in one round (accident guard)")
    ap.add_argument("--dry-run", action="store_true", help="with --escalate: show the calls that would be made; send and write nothing")
    ap.add_argument("--resume", action="store_true", help="continue run folders made by this pipeline version")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing existing output files")
    args = ap.parse_args(argv)
    out_csv = args.out or run_file("results.csv")
    out_final = args.final_out or run_file("final.csv")
    out_json = args.json_out or run_file("decisions.json")
    policy_dir = Path(args.policy_run or run_file("policy"))
    confirm_dir = Path(args.confirm_run or run_file("confirm"))
    mols = load_manifest()
    smiles = graphcheck.load_smiles(args.smiles_csv) if args.smiles_csv else {}
    if not args.dry_run:
        for path in (out_csv, out_final, out_json):
            new_output(path, args.overwrite)
    if args.escalate:
        if not all(prompt_status().values()):
            cli._die("a generated prompt file is missing or does not match chemverify.policy; run `pipeline_google10.py policy --write`")
        if not args.dry_run:
            if not args.evidence_from:
                writable_run_dir(policy_dir, args.resume)
            writable_run_dir(confirm_dir, args.resume)
    evidence = by_task(Path(args.evidence_from)) if args.evidence_from else None
    evidence_source = f"recorded:{Path(args.evidence_from).name}" if args.evidence_from else f"fresh:{policy_dir.name}"
    made = {"policy_calls": 0, "policy_usd": 0.0, "confirm_calls": 0, "confirm_usd": 0.0}
    if args.escalate:
        made = escalate(args, mols, policy_dir, confirm_dir, smiles, evidence)
    if args.dry_run:
        print("dry run: nothing was sent and nothing was written")
        return 0
    policy_recs = evidence if evidence is not None else (by_task(policy_dir) if (policy_dir / "outputs.jsonl").exists() else {})
    confirm_recs = by_task(confirm_dir) if (confirm_dir / "outputs.jsonl").exists() else {}
    decisions = compute_decisions(mols, policy_recs, confirm_recs, smiles, final=True)
    ledger = cost_ledger(mols, policy_recs, confirm_recs)
    rows = build_rows(mols, decisions, ledger, evidence_source)
    write_csv(out_csv, COLUMNS, rows)
    write_csv(out_final, FINAL_COLUMNS, rows)
    prompts = {"policy": prompt_sha(POLICY_PROMPT), "confirm": prompt_sha(CONFIRM_PROMPT)}
    Path(out_json).write_text(json.dumps({
        "schema_version": policy.SCHEMA_VERSION, "policy_version": policy.POLICY_VERSION, "decision_version": policy.DECISION_VERSION,
        "evidence_source": evidence_source, "prompt_sha": prompts,
        "cost": {**ledger["totals"], "made_in_this_invocation": {k: round(v, 4) for k, v in made.items()}, "per_pair": ledger["per_pair"]},
        "decisions": {k: v.to_dict() for k, v in decisions.items()}}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{'task':13s} {'outcome':10s} {'rule':26s} {'evidence':14s} {'policy':>6s} {'confirm':>7s}  failure categories / causes")
    for m in mols:
        d, c = decisions[m["id"]], ledger["per_pair"][m["id"]]
        print(f"{m['id']:13s} {d.outcome:10s} {d.rule:26s} {d.evidence_status:14s} {c['policy_calls']:>6d} {c['confirm_calls']:>7d}  {'; '.join(d.failure_categories)}")
    outcomes = Counter(d.outcome for d in decisions.values())
    t = ledger["totals"]
    print(f"\noutcomes: {dict(outcomes)} | consistency conflicts: {sum(1 for d in decisions.values() if d.consistency_status == 'conflict')}")
    print(f"API calls and cost by stage (everything recorded for this output):\n"
          f"  policy-prompt calls   {t['policy_calls']:>4d}  ${t['policy_cost_usd']:.4f}   source: {evidence_source}\n"
          f"  confirming calls      {t['confirm_calls']:>4d}  ${t['confirm_cost_usd']:.4f}   (the additional cost of the confirmation step)\n"
          f"  total                 {t['policy_calls'] + t['confirm_calls']:>4d}  ${t['total_cost_usd']:.4f}\n"
          f"  made in this invocation: {made['policy_calls']} policy / {made['confirm_calls']} confirming calls, "
          f"${made['policy_usd'] + made['confirm_usd']:.4f}")
    print(f"written: {out_csv}\n         {out_final}\n         {out_json}")
    return 0


# --------------------------------------------------------------------------- metrics (optional, needs labels)


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
    ap.add_argument("--results", default=None)
    ap.add_argument("--labels", default=str(DATA10 / "provisional_labels.csv"), help="CSV with IDs and correct = yes / no / unknown")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    results = args.results or run_file("results.csv")
    out = args.out or run_file("metrics.csv")
    with open(args.labels, newline="", encoding="utf-8-sig") as fh:
        labels = {r["IDs"].strip(): r["correct"].strip().lower() for r in csv.DictReader(fh)}
    with open(results, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    known = [r for r in rows if labels.get(r["IDs"]) in ("yes", "no")]
    skipped = [r["IDs"] for r in rows if labels.get(r["IDs"]) not in ("yes", "no")]
    result = confusion([(r.get("final_outcome") == "ACCEPT", labels[r["IDs"]] == "yes") for r in known])
    print(f"decision layer (final_outcome)   (n={result['n']}, left out: {', '.join(skipped) or 'none'})")
    print(f"{'':32s}{'actually correct':>18s}{'actually wrong':>16s}")
    print(f"{'accepted':32s}{'TP = ' + str(result['TP']):>18s}{'FP = ' + str(result['FP']):>16s}")
    print(f"{'not accepted (reject/uncertain)':32s}{'FN = ' + str(result['FN']):>18s}{'TN = ' + str(result['TN']):>16s}")
    print("precision {precision}  recall {recall}  F1 {f1}  specificity {specificity}  accuracy {accuracy}".format(**result))
    write_csv(new_output(out, False), ["system", "n", "TP", "FP", "FN", "TN", "precision", "recall", "f1", "specificity", "accuracy", "left_out"],
              [{"system": "decision layer (final_outcome)", **result, "left_out": "; ".join(skipped)}])
    print(f"written: {out}")
    return 0


def main(argv: list[str]) -> int:
    commands = {"prep": cmd_prep, "policy": cmd_policy, "decide": cmd_decide, "metrics": cmd_metrics}
    argv = list(argv)
    for flag, apply in (("--set", use_dataset), ("--run-tag", use_tag)):  # global options, accepted anywhere before the command
        if flag in argv:
            position = argv.index(flag)
            if position + 1 >= len(argv):
                cli._die(f"{flag} needs a value")
            apply(argv[position + 1])
            del argv[position:position + 2]
    if not argv or argv[0] not in commands:
        print(__doc__)
        return 2
    return commands[argv[0]](argv[1:])


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main(sys.argv[1:]))
