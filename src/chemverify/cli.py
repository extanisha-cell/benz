"""Command-line interface:  python -I run.py <command> --help"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import gemini, gold, metrics, openrouter, perception, report, runner, splits
from . import pairs as pairs_mod
from .download import download_all
from .manifest import ALLOWED_HOST_SUFFIXES, read_rows, sanitize_url
from .paths import DATA, GOLD, HELDOUT_LOG, IMAGES, MANIFEST, ROLES, ROOT, RUNS, SPLITS
from .prompts import load_prompt

PROBE_INSTRUCTION = "Transcribe this drawing."


def _die(message: str):
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _err(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _load_manifest() -> dict:
    if not MANIFEST.exists():
        _die("data/manifest.json not found; run `prep` first")
    return json.loads(MANIFEST.read_text(encoding="utf-8-sig"))


def _gold_primary() -> dict[str, str] | None:
    path = GOLD / "errors.json"
    if not path.exists():
        return None
    return {m: e["primary"] for m, e in json.loads(path.read_text(encoding="utf-8-sig")).items()}


def _gold_correct_issues() -> dict[str, list[dict]] | None:
    """molecule -> issues the gold agents recorded in its corrected drawing (only molecules that have any)."""
    path = GOLD / "errors.json"
    if not path.exists():
        return None
    errors = json.loads(path.read_text(encoding="utf-8-sig"))
    return {m: e["correct_version_issues"] for m, e in errors.items() if e.get("correct_version_issues")}


def _backend(args) -> str:
    """--backend if given; otherwise whatever the stored key belongs to (Google when there is no key yet)."""
    requested = getattr(args, "backend", "auto")
    if requested != "auto":
        return requested
    key, source = gemini.get_api_key()
    return gemini.detect_backend(key, source)


def _settings(args) -> gemini.GenSettings:
    backend = _backend(args)
    default_model = openrouter.DEFAULT_MODEL if backend == "openrouter" else gemini.DEFAULT_MODEL
    try:
        return gemini.GenSettings(
            model=args.model or default_model,
            temperature=args.temperature,
            thinking_level=args.thinking_level,
            media_resolution=args.media_resolution,
            max_output_tokens=args.max_output_tokens,
            seed_base=args.seed_base,
            backend=backend,
            route_to=tuple(args.route_to) if args.route_to else None,
            data_collection=args.data_collection,
        )
    except ValueError as exc:
        _die(str(exc))


def _client(settings: gemini.GenSettings, args) -> gemini.GeminiClient:
    key, source = gemini.get_api_key()
    if not key:
        _die(f"no API key found. Put OPENROUTER_API_KEY=<key> (or GEMINI_API_KEY=<key>) in {gemini.ENV_FILE} (see README). "
             "The key is never printed or stored.")
    _err(f"using API key from {source}; backend: {settings.backend}; model: {settings.model}")
    if (hint := gemini.key_shape_warning(key, settings.backend)):
        _err(f"warning: {hint}")
    if settings.backend == "openrouter":
        _err("note: images go to openrouter.ai, which forwards them to the model provider "
             f"(provider routing: {', '.join(settings.route_to) if settings.route_to else 'chosen by OpenRouter'}; "
             f"data_collection: {settings.data_collection or 'your OpenRouter account default'})")
        return openrouter.OpenRouterClient(key, settings, rpm=args.rpm)
    return gemini.GeminiClient(key, settings, rpm=args.rpm)


def _check_budget(args, settings: gemini.GenSettings) -> None:
    if args.budget_usd is not None and settings.backend != "openrouter":
        _die("--budget-usd needs the openrouter backend: Google's own API does not report the cost of a call, "
             "so the budget could never trigger")


def _build_request(prompt_text: str, parts: list, settings: gemini.GenSettings) -> dict:
    module = openrouter if settings.backend == "openrouter" else gemini
    return module.build_request(prompt_text, parts, settings)


# --------------------------------------------------------------------------- commands


def cmd_status(args) -> int:
    print(f"python {sys.version.split()[0]}   project {ROOT}")
    key, source = gemini.get_api_key()
    print(f"API key: {'present' if key else 'NOT set'} ({source})")
    if key:
        backend = gemini.detect_backend(key, source)
        default_model = openrouter.DEFAULT_MODEL if backend == "openrouter" else gemini.DEFAULT_MODEL
        print(f"backend: {backend} (default model {default_model}; override with --backend / --model)")
        if (hint := gemini.key_shape_warning(key, backend)):
            print(f"  warning: {hint}")
    if not gemini.ENV_FILE.exists():
        file_state = "not created"
    else:
        file_state = "has a key" if any(gemini.read_env_file(gemini.ENV_FILE).get(n) for n in gemini.KEY_VARS) else "exists, no key in it yet"
    print(f"key file: {gemini.ENV_FILE} ({file_state})")
    if MANIFEST.exists():
        mols = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))["molecules"]
        print(f"manifest: {len(mols)} molecules, {sum(m['complete'] for m in mols)} complete (reference + wrong + correct)")
    else:
        print("manifest: not built yet (run `prep`)")
    split = splits.load_split(SPLITS)
    print(f"split: {len(split['dev'])} dev / {len(split['heldout'])} held-out (frozen)" if split else "split: not created yet")
    print(f"gold labels: {'merged' if (GOLD / 'errors.json').exists() else 'not merged yet'}")
    if RUNS.exists():
        for run in sorted(p for p in RUNS.iterdir() if (p / "meta.json").exists()):
            n = len(runner.load_records(run / "outputs.jsonl"))
            print(f"run {run.name}: {n} records")
    return 0


def cmd_prep(args) -> int:
    rows = read_rows(args.raw_csv)
    allowed = ALLOWED_HOST_SUFFIXES + tuple(args.allow_host or ())
    manifest, problems = download_all(rows, IMAGES, workers=args.workers, force=args.force, allowed_hosts=allowed, log=_err)
    DATA.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with open(DATA / "samples.csv", "w", newline="", encoding="utf-8") as fh:  # signed query strings removed
        writer = csv.writer(fh)
        writer.writerow(["IDs", "Samples", "Wrongly annotated", "Correct version"])
        for row in rows:
            writer.writerow([row.id] + [sanitize_url(row.url(r)) for r in ROLES])
    print(f"\n{'molecule':14s} {'role':8s} {'format':6s} {'size':>11s} {'bytes':>8s}  notes")
    for mol in manifest["molecules"]:
        for role, e in mol["images"].items():
            notes = []
            if e["has_alpha"]:
                notes.append("has transparency")
            if not e["complete"]:
                notes.append("looks truncated")
            if not e["gemini_supported"]:
                notes.append("NOT accepted by Gemini")
            size = f"{e['width']}x{e['height']}" if e["width"] else "?"
            print(f"{mol['id']:14s} {role:8s} {e['format']:6s} {size:>11s} {e['bytes']:>8d}  {', '.join(notes)}")
    complete = sum(m["complete"] for m in manifest["molecules"])
    print(f"\n{complete}/{len(manifest['molecules'])} molecules complete; manifest written to {MANIFEST}")
    if problems:
        print("\nproblems:")
        for p in problems:
            print(f"  - {p}")
        return 1
    return 0


def cmd_models(args) -> int:
    key, _ = gemini.get_api_key()
    if not key:
        _die(f"no API key found; put it in {gemini.ENV_FILE} first (see README)")
    module = openrouter if _backend(args) == "openrouter" else gemini
    models = module.list_models(key)
    for m in sorted(models, key=lambda x: x["name"]):
        if args.all or ("gemini" in m["name"] and any(x in m["methods"] for x in ("generateContent", "chat.completions"))):
            print(f"{m['name']:44s} {m['display_name'] or ''}")
    return 0


def cmd_split(args) -> int:
    images = pairs_mod.complete_images(_load_manifest())
    primary = None
    if args.stratify_gold:
        primary = _gold_primary()
        if primary is None:
            _die("no data/gold/errors.json yet; finish the gold labelling and run `gold merge` first")
    split = splits.make_split(sorted(images), n_heldout=args.heldout, seed=args.seed, primary=primary)
    try:
        splits.save_split(split, SPLITS, force=args.force)
    except FileExistsError as exc:
        _die(str(exc))
    print(json.dumps(split, indent=2))
    return 0


def _select_pairs(args, images: dict, split: dict | None):
    try:
        molecules = pairs_mod.molecules_for(args.split, images, split)
    except ValueError as exc:
        _die(str(exc))
    include = tuple(x for x in args.include.split(",") if x)
    try:
        selected = pairs_mod.build_pairs(images, molecules, include=include, identity_n=args.identity_n, seed=args.pair_seed)
    except ValueError as exc:
        _die(str(exc))
    return molecules, selected


def cmd_pairs(args) -> int:
    images = pairs_mod.complete_images(_load_manifest())
    _, selected = _select_pairs(args, images, splits.load_split(SPLITS))
    for p in selected:
        print(f"{p.id:44s} expected={p.expected} {p.category:18s} gated={p.gated}")
    print(f"{len(selected)} pairs")
    return 0


def _dry_run(tasks, prompt, settings, runs: int) -> None:
    for task in tasks:
        body = _build_request(prompt.text, task.load_parts(), settings)  # also checks the images
        print(f"{task.id:48s} {len(json.dumps(body)) / 1024:8.1f} KiB request")
    print(f"{len(tasks)} tasks x {runs} runs = {len(tasks) * runs} calls (dry run: nothing was sent)")


def _write_pair_report(run_dir: Path, *, show_heldout: bool = False, localisation_path: str | None = None) -> Path:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8-sig"))
    records = list(runner.load_records(run_dir / "outputs.jsonl").values())
    split = splits.load_split(SPLITS)
    spend = runner.total_cost(records)  # a dollar figure says nothing about which pairs passed, so held-out calls count too
    excluded = 0
    if not show_heldout:  # held-out pairs are dropped before any number is computed, so totals cannot leak them
        records, excluded = splits.drop_heldout(records, split)

    def split_of(molecule: str) -> str:
        return splits.split_of(molecule, split)

    result = metrics.compute(records, split_of=split_of, gold_primary=_gold_primary(), correct_issues=_gold_correct_issues())
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    localisation = json.loads(Path(localisation_path).read_text(encoding="utf-8-sig")) if localisation_path else None
    versions = Counter(r.get("model_version") or "unreported" for r in records if r.get("status") == "ok")
    text = report.render(meta, result, show_heldout=show_heldout, split_of=split_of, localisation=localisation,
                         heldout_excluded=excluded, model_versions=dict(versions), spend=spend)
    out = run_dir / "report.md"
    out.write_text(text, encoding="utf-8")
    return out


def _run_meta(args, kind: str, prompt, settings, **extra) -> dict:
    return {
        "name": args.name,
        "kind": kind,
        "prompt_name": prompt.name,
        "prompt_sha": prompt.sha,
        "format": args.format,
        "settings": settings.to_dict(),
        "settings_fp": settings.fingerprint(),
        "runs": args.runs,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }


def cmd_eval(args) -> int:
    prompt = load_prompt(args.prompt)
    images = pairs_mod.complete_images(_load_manifest())
    split = splits.load_split(SPLITS)
    molecules, selected = _select_pairs(args, images, split)
    heldout = pairs_mod.touches_heldout(molecules, split)
    if heldout and not args.allow_heldout:
        _die("this selection includes held-out molecules. Held-out data is for the final test only; "
             "pass --allow-heldout to override (the access is logged).")
    if args.only:
        selected = [p for p in selected if p.id in set(args.only)]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        _die("no pairs selected")
    tasks = runner.pair_tasks(selected, DATA, labels=(args.labels == "on"))
    settings = _settings(args)
    _check_budget(args, settings)
    if args.dry_run:
        _dry_run(tasks, prompt, settings, args.runs)
        return 0
    run_dir = RUNS / args.name
    meta = _run_meta(args, "pairs", prompt, settings, split=args.split, include=args.include, labels=args.labels)
    client = _client(settings, args)
    try:
        runner.init_run_dir(run_dir, meta, prompt)
    except runner.RunMismatch as exc:
        _die(str(exc))
    if heldout:
        runner.log_heldout_access(HELDOUT_LOG, args.name, prompt.sha, len(selected))
    try:
        summary = runner.run_tasks(tasks, run_dir=run_dir, prompt=prompt, fmt=args.format, client=client, runs=args.runs,
                                   workers=args.workers, max_calls=args.max_calls, retry_errors=not args.no_retry_errors,
                                   budget_usd=args.budget_usd)
    except runner.BudgetExceeded as exc:
        _die(str(exc))
    except KeyboardInterrupt:
        return 130
    print(f"finished: {summary}")
    print(f"report: {_write_pair_report(run_dir)}")
    return 0


def cmd_probe(args) -> int:
    prompt = load_prompt(args.prompt)
    images = pairs_mod.complete_images(_load_manifest())
    tasks = runner.image_tasks(images, DATA, sorted(images), PROBE_INSTRUCTION)
    if args.limit:
        tasks = tasks[: args.limit]
    args.format = "perception_json"
    settings = _settings(args)
    _check_budget(args, settings)
    if args.dry_run:
        _dry_run(tasks, prompt, settings, args.runs)
        return 0
    run_dir = RUNS / args.name
    client = _client(settings, args)
    try:
        runner.init_run_dir(run_dir, _run_meta(args, "perception", prompt, settings), prompt)
    except runner.RunMismatch as exc:
        _die(str(exc))
    try:
        summary = runner.run_tasks(tasks, run_dir=run_dir, prompt=prompt, fmt="perception_json", client=client, runs=args.runs,
                                   workers=args.workers, max_calls=args.max_calls, retry_errors=not args.no_retry_errors,
                                   budget_usd=args.budget_usd)
    except runner.BudgetExceeded as exc:
        _die(str(exc))
    except KeyboardInterrupt:
        return 130
    print(f"finished: {summary}")
    print(f"score it with:  python -I run.py score-perception --run runs/{args.name}")
    return 0


def cmd_report(args) -> int:
    for run in args.run:
        run_dir = Path(run)
        if args.show_heldout:
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8-sig"))
            runner.log_heldout_access(HELDOUT_LOG, run_dir.name, meta["prompt_sha"], 0)
        out = _write_pair_report(run_dir, show_heldout=args.show_heldout, localisation_path=args.localisation)
        if len(args.run) == 1:
            print(out.read_text(encoding="utf-8-sig"))
        else:
            print(out)
    return 0


def cmd_compare(args) -> int:
    split = splits.load_split(SPLITS)
    rows = []
    for run in args.run:
        run_dir = Path(run)
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8-sig"))
        if args.show_heldout:
            runner.log_heldout_access(HELDOUT_LOG, run_dir.name, meta["prompt_sha"], 0)
        records = list(runner.load_records(run_dir / "outputs.jsonl").values())
        if not args.show_heldout:
            records, _ = splits.drop_heldout(records, split)
        rows.append((meta, metrics.compute(records, split_of=lambda m: splits.split_of(m, split))))
    print(report.render_compare(rows))
    return 0


def cmd_score_perception(args) -> int:
    run_dir = Path(args.run)
    gold_path = Path(args.gold)
    if not gold_path.exists():
        _die(f"{gold_path} not found; finish gold labelling and run `gold merge` first")
    gold_t = json.loads(gold_path.read_text(encoding="utf-8-sig"))
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8-sig"))
    scored, skipped = [], 0
    for rec in runner.load_records(run_dir / "outputs.jsonl").values():
        reference = gold_t.get(rec.get("image_key", ""))
        if reference is None or rec.get("status") != "ok":
            skipped += 1
            continue
        s = perception.score(rec.get("data"), reference)
        s.update(image_key=rec["image_key"], role=rec["role"], molecule=rec["molecule"], run_idx=rec["run_idx"])
        scored.append(s)
    if not scored:
        _die("nothing to score (no successful probe runs with matching gold labels)")
    summary = perception.summarise(scored)
    (run_dir / "perception_scores.json").write_text(json.dumps({"summary": summary, "scored": scored}, indent=2) + "\n", encoding="utf-8")
    text = perception.render(summary, meta)
    (run_dir / "perception_report.md").write_text(text, encoding="utf-8")
    print(text)
    if skipped:
        print(f"({skipped} records skipped: failed calls or no gold label)")
    return 0


def cmd_gold(args) -> int:
    if args.gold_cmd == "validate":  # lets a labelling agent check its own file; needs no manifest
        try:
            obj = json.loads(Path(args.file).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            print(f"cannot read {args.file}: {exc}")
            return 1
        if args.kind == "transcriptions":
            tokens = None
            if args.batch:  # a batch file lists the tokens the transcriber was given
                tokens = [t["token"] for t in json.loads(Path(args.batch).read_text(encoding="utf-8-sig"))["images"]]
            problems = gold.validate_transcriptions(obj, tokens)
        elif not args.molecule:
            print("--molecule is required for this kind")
            return 1
        else:
            check = gold.validate_molecule if args.kind == "adjudication" else gold.validate_verification
            problems = check(obj, args.molecule)
        print("OK" if not problems else "PROBLEMS:\n- " + "\n- ".join(problems))
        return 1 if problems else 0
    images = pairs_mod.complete_images(_load_manifest())
    if args.gold_cmd == "prepare-blind":
        tasks = gold.prepare_blind(images, DATA, GOLD)
        (GOLD / "blind_tasks.json").write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
        print(f"{len(tasks)} blind copies written under {GOLD / 'blind'}")
        return 0
    if args.gold_cmd == "csv":
        counts = gold.export_csv(GOLD / "by_molecule", Path(args.out), sorted(images))
        print(f"{args.out}: {counts}")
        return 0
    summary = gold.merge(GOLD / "by_molecule", GOLD, sorted(images))
    print(json.dumps(summary, indent=2))
    print(f"review table: {GOLD / 'review_table.md'}")
    return 0 if not summary["invalid"] and not summary["missing"] else 1


def cmd_test(args) -> int:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


# --------------------------------------------------------------------------- parser


def _add_backend_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=("auto",) + gemini.BACKENDS, default="auto",
                   help="who to call: auto picks from the key (sk-or-... = openrouter, otherwise Google directly)")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    _add_backend_arg(p)
    p.add_argument("--model", default=None, help="default: gemini-3.1-pro-preview on Google, google/gemini-3.1-pro-preview on OpenRouter")
    p.add_argument("--route-to", nargs="+", metavar="SLUG", default=None,
                   help="openrouter only: provider slug(s) to use, with no fallback to others (copy the slug from the model page)")
    p.add_argument("--data-collection", choices=gemini.DATA_COLLECTION, default=None,
                   help="openrouter only: 'deny' uses only providers that do not keep or train on your data (default: your account setting)")
    p.add_argument("--thinking-level", choices=gemini.THINKING_LEVELS, default=None, help="default: API default (high on 3.1 Pro)")
    p.add_argument("--media-resolution", choices=gemini.MEDIA_RESOLUTIONS, default=None, help="default: API default")
    p.add_argument("--temperature", type=float, default=None, help="default: API default (Google recommends 1.0 for Gemini 3)")
    p.add_argument("--max-output-tokens", type=int, default=None)
    p.add_argument("--seed-base", type=int, default=None, help="run i uses seed base+i; default: no seed")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rpm", type=float, default=None, help="cap on requests per minute")
    p.add_argument("--max-calls", type=int, default=500, help="refuse to start more calls than this")
    p.add_argument("--limit", type=int, default=None, help="only the first N tasks (smoke test)")
    p.add_argument("--dry-run", action="store_true", help="build requests and check images; send nothing")
    p.add_argument("--no-retry-errors", action="store_true")
    p.add_argument("--budget-usd", type=float, default=None,
                   help="stop starting new calls once this much has been spent in this invocation (openrouter reports cost; "
                        "calls already in flight still finish, so the total can overshoot by up to (--workers - 1) calls)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run.py", description="Score system prompts for chemical-structure verification.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="show key, data, split and runs").set_defaults(func=cmd_status)

    p = sub.add_parser("prep", help="download the images listed in the samples CSV")
    p.add_argument("--raw-csv", required=True, help="CSV with Drive/S3 links (keep it outside the project: it can hold signed URLs)")
    p.add_argument("--force", action="store_true", help="re-download even if files exist")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--allow-host", action="append", help="extra host suffix to allow")
    p.set_defaults(func=cmd_prep)

    p = sub.add_parser("models", help="list models visible to your API key")
    p.add_argument("--all", action="store_true")
    _add_backend_arg(p)
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("split", help="create the frozen dev / held-out split")
    p.add_argument("--heldout", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--stratify-gold", action="store_true", help="spread error categories across the held-out set")
    p.add_argument("--force", action="store_true", help="replace an existing split (invalidates earlier results)")
    p.set_defaults(func=cmd_split)

    def pair_args(q: argparse.ArgumentParser) -> None:
        q.add_argument("--split", choices=("dev", "heldout", "all"), default="dev")
        q.add_argument("--include", default="annotated,cross", help="comma list of: annotated, cross, identity")
        q.add_argument("--identity-n", type=int, default=5)
        q.add_argument("--pair-seed", type=int, default=7)

    p = sub.add_parser("pairs", help="list the pairs a selection would use")
    pair_args(p)
    p.set_defaults(func=cmd_pairs)

    p = sub.add_parser("eval", help="run a prompt over labelled pairs and report accuracy")
    p.add_argument("--prompt", required=True)
    p.add_argument("--format", choices=("number", "auditor_json", "verdict_json"), default="number")
    p.add_argument("--name", required=True, help="run folder name under runs/")
    p.add_argument("--runs", type=int, default=5, help="repeats per pair")
    p.add_argument("--labels", choices=("on", "off"), default="on", help="send 'Image A/B' text labels with the images")
    p.add_argument("--only", nargs="+", help="restrict to these pair ids")
    p.add_argument("--allow-heldout", action="store_true", help="final test only; the access is logged")
    pair_args(p)
    _add_model_args(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("probe", help="perception probe: transcribe each image to a fixed schema")
    p.add_argument("--prompt", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--runs", type=int, default=3)
    _add_model_args(p)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("score-perception", help="score a probe run against the gold transcriptions")
    p.add_argument("--run", required=True)
    p.add_argument("--gold", default=str(GOLD / "transcriptions.json"))
    p.set_defaults(func=cmd_score_perception)

    p = sub.add_parser("report", help="(re)build the report for one or more runs")
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--show-heldout", action="store_true", help="include held-out pairs (final test only; logged)")
    p.add_argument("--localisation", help="JSON {pair_id: true|false} from the judge step")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("compare", help="side-by-side table of runs")
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--show-heldout", action="store_true", help="include held-out pairs (final test only; logged)")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("gold", help="gold-label helpers")
    gsub = p.add_subparsers(dest="gold_cmd", required=True)
    gsub.add_parser("prepare-blind", help="copy images to neutral names for blind transcription")
    gsub.add_parser("merge", help="validate per-molecule gold files and build errors.json / review table")
    c = gsub.add_parser("csv", help="one row per molecule from the adjudicators' files, as a CSV (opens in Excel)")
    c.add_argument("--out", default=str(GOLD / "gold_labels.csv"))
    v = gsub.add_parser("validate", help="check one agent-written gold file against its schema")
    v.add_argument("--kind", choices=("transcriptions", "adjudication", "verification"), required=True)
    v.add_argument("--file", required=True)
    v.add_argument("--molecule", help="needed for adjudication and verification files")
    v.add_argument("--batch", help="transcriptions only: the batch file listing the images you were given")
    p.set_defaults(func=cmd_gold)

    p = sub.add_parser("test", help="run the unit tests")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except gemini.GeminiError as exc:
        _err(f"error: {exc}")
        return 1
