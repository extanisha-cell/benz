"""Run a prompt over tasks (image pairs or single images) and persist every raw output.

Results go to runs/<name>/outputs.jsonl, one JSON object per (task, run). A re-run skips
completed items and retries failed ones, so an interrupted or rate-limited run can simply be
started again. Images are sent exactly as downloaded (original bytes and MIME type), which
mirrors what a platform that passes image URLs through would feed the model.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .gemini import GeminiClient, GeminiError, ImagePart, TextPart, redact
from .imageinfo import GEMINI_MIME_TYPES, sniff
from .parse import PARSERS
from .prompts import PromptSpec


class RunMismatch(Exception):
    """The run folder already holds results produced with a different prompt or settings."""


class BudgetExceeded(Exception):
    pass


@dataclass
class Task:
    id: str
    meta: dict
    load_parts: Callable[[], list]
    input_fp: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _read_image(path: Path) -> ImagePart:
    data = path.read_bytes()
    info = sniff(data)
    if info is None or info.mime not in GEMINI_MIME_TYPES:
        raise ValueError(f"{path.name}: not an image type Gemini accepts")
    return ImagePart(data, info.mime)


def pair_tasks(pairs, data_root: Path, *, labels: bool = True) -> list[Task]:
    """One task per pair. Image A is always sent first, image B second."""
    tasks: list[Task] = []
    for pair in pairs:

        def loader(pair=pair) -> list:
            parts: list = []
            if labels:
                parts.append(TextPart("Image A (reference chemical structure):"))
            parts.append(_read_image(data_root / pair.a))
            if labels:
                parts.append(TextPart("Image B (annotator's handwritten reproduction):"))
            parts.append(_read_image(data_root / pair.b))
            return parts

        fp = hashlib.sha256(f"{pair.a_sha}|{pair.b_sha}|labels={labels}".encode()).hexdigest()[:12]
        tasks.append(Task(pair.id, {"kind": "pair", **pair.meta()}, loader, fp))
    return tasks


def image_tasks(images: dict, data_root: Path, molecules: list[str], instruction: str) -> list[Task]:
    """One task per single image, for the perception probe."""
    tasks: list[Task] = []
    for m in molecules:
        for role, entry in images[m].items():

            def loader(entry=entry) -> list:
                return [TextPart(instruction), _read_image(data_root / entry["file"])]

            meta = {"kind": "image", "image_key": f"{m}/{role}", "molecule": m, "role": role, "sha": entry["sha256"][:12]}
            tasks.append(Task(f"{m}/{role}", meta, loader, entry["sha256"][:12]))
    return tasks


def record_key(prompt: PromptSpec, settings_fp: str, task: Task, run_idx: int) -> str:
    raw = f"{prompt.sha}|{settings_fp}|{task.id}|{task.input_fp}|{run_idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def load_records(path: Path) -> dict[str, dict]:
    """Latest record per key. Tolerates a torn final line left by an interrupted run."""
    records: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                records[rec["key"]] = rec
    return records


def init_run_dir(run_dir: Path, meta: dict, prompt: PromptSpec) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        for field in ("kind", "prompt_sha", "settings_fp", "format"):
            if old.get(field) != meta.get(field):
                raise RunMismatch(
                    f"{run_dir.name} was created with {field}={old.get(field)!r}, now {meta.get(field)!r}. "
                    "Use a new --name so results from different prompts or settings are never mixed."
                )
        return
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (run_dir / "prompt.txt").write_text(prompt.text + "\n", encoding="utf-8")


def run_tasks(
    tasks: list[Task],
    *,
    run_dir: Path,
    prompt: PromptSpec,
    fmt: str,
    client: GeminiClient,
    runs: int,
    workers: int = 4,
    max_calls: int = 500,
    retry_errors: bool = True,
    budget_usd: float | None = None,
    log: Callable[[str], None] = _log,
) -> dict:
    parser = PARSERS[fmt]
    out_path = run_dir / "outputs.jsonl"
    existing = load_records(out_path)
    settings_fp = client.settings.fingerprint()
    todo = []
    for idx in range(runs):  # run-major: every pair gets run 0 before any pair gets run 1, so a budget stop leaves
        for task in tasks:  # all pairs evenly covered instead of the last pairs not evaluated at all
            key = record_key(prompt, settings_fp, task, idx)
            prev = existing.get(key)
            if prev is not None and (prev.get("status") == "ok" or not retry_errors):
                continue
            todo.append((task, idx, key))
    total = len(tasks) * runs
    log(f"{len(tasks)} tasks x {runs} runs = {total} calls; {total - len(todo)} already done, {len(todo)} to run")
    if len(todo) > max_calls:
        raise BudgetExceeded(f"{len(todo)} calls exceeds --max-calls {max_calls}; raise the limit if that is intended")

    write_lock = threading.Lock()
    counter = {"done": 0}
    budget = {"spent": 0.0, "hit": False}  # updated by the workers themselves, so no call starts after the budget is reached

    def work(item) -> dict | None:
        task, idx, key = item
        with write_lock:
            if budget["hit"]:
                return None  # budget reached: this call is never started
        started = time.monotonic()
        rec = {
            "key": key, "task_id": task.id, "run_idx": idx, **task.meta,
            "prompt": prompt.name, "prompt_sha": prompt.sha, "model": client.settings.model,
            "settings_fp": settings_fp, "ts": _now(),
        }
        try:
            result = client.generate(prompt.text, task.load_parts(), run_idx=idx)
            rec.update(finish_reason=result.finish_reason, usage=result.usage,
                       response_id=result.response_id, model_version=result.model_version)
            if result.block_reason:
                rec.update(status="blocked", error=f"blocked: {result.block_reason}")
            elif not result.text.strip():
                rec.update(status="empty", error=f"empty response (finishReason={result.finish_reason})")
            else:
                parsed = parser(result.text)
                rec.update(status="ok", raw_text=result.text, decision=parsed.decision, reason=parsed.reason,
                           valid=parsed.valid, problems=parsed.problems,
                           lenient_decision=parsed.lenient_decision, data=parsed.data)
        except GeminiError as exc:
            rec.update(status="error", error=str(exc))
        except Exception as exc:  # keep the run going; the record carries the failure
            rec.update(status="error", error=redact(f"{type(exc).__name__}: {exc}", None))
        rec["latency_s"] = round(time.monotonic() - started, 2)
        with write_lock:
            budget["spent"] += float((rec.get("usage") or {}).get("cost") or 0)
            if budget_usd is not None and budget["spent"] >= budget_usd and not budget["hit"]:
                budget["hit"] = True  # calls already in flight still finish and are saved; nothing new starts
                log(f"budget of ${budget_usd:.2f} reached (${budget['spent']:.2f} spent): no new calls will start. Re-run to continue.")
        return rec

    executor = cf.ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [executor.submit(work, item) for item in todo]
        with open(out_path, "a", encoding="utf-8") as sink:
            for future in cf.as_completed(futures):
                rec = future.result()
                if rec is None:
                    continue
                with write_lock:
                    sink.write(json.dumps(rec) + "\n")
                    sink.flush()
                    counter["done"] += 1
                    shown = rec.get("decision") if rec["status"] == "ok" else rec["status"]
                    log(f"[{counter['done']}/{len(todo)}] {rec['task_id']} run {rec['run_idx']} -> {shown} ({rec['latency_s']}s)")
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        log("interrupted: completed calls are saved; re-run the same command to resume")
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    final = load_records(out_path)
    statuses = Counter(r.get("status") for r in final.values())
    summary = {"records": len(final), "statuses": dict(statuses), "spent_usd": round(budget["spent"], 4)}
    if len(todo) > counter["done"]:  # only possible when the budget stopped the run
        summary["stopped_by_budget"] = True
        summary["calls_not_started"] = len(todo) - counter["done"]
    return summary


def total_cost(records) -> dict:
    """Cost the API reported across records (OpenRouter reports it; Google's own API does not)."""
    costs = [float((r.get("usage") or {}).get("cost")) for r in records if (r.get("usage") or {}).get("cost") is not None]
    return {"usd": round(sum(costs), 4), "calls": len(costs)}


def log_heldout_access(path: Path, run_name: str, prompt_sha: str, n_pairs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{_now()}\trun={run_name}\tprompt={prompt_sha}\tpairs={n_pairs}\n")
