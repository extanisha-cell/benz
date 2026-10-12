"""Perception probe on the 10 single reference images in data/tasks10/ (no handwritten Image B exists for them).

Mirrors `run.py probe`, but reads data/tasks10/*.png instead of data/manifest.json, so the 17-molecule
manifest, split and gold files are untouched.

    python -I probe_tasks10.py --prompt prompts/perception_probe.txt --name tasks10_probe --runs 1 \
        --thinking-level low --budget-usd 1 [--dry-run]
"""
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from chemverify import cli, runner  # noqa: E402
from chemverify.prompts import load_prompt  # noqa: E402

DATA10 = ROOT / "data" / "tasks10"


def main(argv: list[str]) -> int:
    args = cli.build_parser().parse_args(["probe", *argv])
    prompt = load_prompt(args.prompt)
    files = sorted(DATA10.glob("molecule_*.png"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
    if not files:
        cli._die(f"no images in {DATA10}")
    images = {p.stem: {"ref": {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}} for p in files}
    tasks = runner.image_tasks(images, DATA10, list(images), cli.PROBE_INSTRUCTION)
    args.format = "perception_json"
    settings = cli._settings(args)
    cli._check_budget(args, settings)
    if args.dry_run:
        cli._dry_run(tasks, prompt, settings, args.runs)
        return 0
    run_dir = cli.RUNS / args.name
    client = cli._client(settings, args)
    try:
        runner.init_run_dir(run_dir, cli._run_meta(args, "perception", prompt, settings), prompt)
        summary = runner.run_tasks(tasks, run_dir=run_dir, prompt=prompt, fmt="perception_json", client=client,
                                   runs=args.runs, workers=args.workers, max_calls=args.max_calls,
                                   retry_errors=not args.no_retry_errors, budget_usd=args.budget_usd)
    except (runner.RunMismatch, runner.BudgetExceeded) as exc:
        cli._die(str(exc))
    print(f"finished: {summary}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main(sys.argv[1:]))
