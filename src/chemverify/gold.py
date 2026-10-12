"""Gold labels: blind copies for independent transcription, and validation / merging of the files
written by the labelling agents.

Two agents author two separate files per molecule, so no file is ever edited by a second writer:
  data/gold/by_molecule/<id>.json   adjudicator: final transcriptions + the errors in the wrong version
  data/gold/verification/<id>.json  independent verifier that tried to refute the adjudicator
`merge` combines them. The verification status is recomputed from the evidence, not taken on trust.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from . import perception
from .paths import ROLES
from .taxonomy import ERROR_CATEGORIES

SEVERITIES = ("high", "medium", "low")
CONFIDENCE = ("high", "medium", "low")
CLAIM_VERDICTS = ("confirmed", "refuted", "uncertain")


def prepare_blind(images: dict, data_root: Path, gold_dir: Path, salt: str = "chemverify-gold") -> list[dict]:
    """Copy every image to a neutral file name so a transcriber cannot tell a reference from a
    wrong or corrected drawing by its path. Returns [{token, path, molecule, role}] (the molecule
    and role are for the orchestrator only; agents are given just the path) and writes blind_map.json."""
    blind_dir = Path(gold_dir) / "blind"
    shutil.rmtree(blind_dir, ignore_errors=True)
    blind_dir.mkdir(parents=True, exist_ok=True)
    tasks, mapping = [], {}
    for molecule in sorted(images):
        for role in ROLES:
            entry = images[molecule][role]
            token = hashlib.sha256(f"{salt}|{molecule}|{role}".encode()).hexdigest()[:10]
            target = blind_dir / f"img_{token}{Path(entry['file']).suffix}"
            shutil.copyfile(Path(data_root) / entry["file"], target)
            mapping[token] = {"molecule": molecule, "role": role}
            tasks.append({"token": token, "path": str(target), "molecule": molecule, "role": role})
    (Path(gold_dir) / "blind_map.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    return tasks


def _check_issue_list(obj: dict, key: str, problems: list[str]) -> list:
    items = obj.get(key)
    if not isinstance(items, list):
        problems.append(f"{key} must be a list")
        return []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            problems.append(f"{key}[{i}] must be an object")
            continue
        if item.get("category") not in ERROR_CATEGORIES:
            problems.append(f"{key}[{i}].category must be one of the taxonomy categories")
        if not str(item.get("description", "")).strip():
            problems.append(f"{key}[{i}].description is empty")
        if item.get("severity") not in SEVERITIES:
            problems.append(f"{key}[{i}].severity must be one of {SEVERITIES}")
    return items


def validate_molecule(obj, molecule_id: str) -> list[str]:
    """Problems with the adjudicator's file (empty list = valid)."""
    if not isinstance(obj, dict):
        return ["not a JSON object"]
    problems: list[str] = []
    if obj.get("molecule") != molecule_id:
        problems.append(f"molecule field is {obj.get('molecule')!r}, expected {molecule_id!r}")
    transcriptions = obj.get("transcriptions")
    if not isinstance(transcriptions, dict):
        problems.append("transcriptions must be an object keyed by ref/wrong/correct")
    else:
        for role in ROLES:
            if role not in transcriptions:
                problems.append(f"transcriptions.{role} is missing")
            else:
                problems += [f"transcriptions.{role}: {p}" for p in perception.validate(transcriptions[role])]
    errors = _check_issue_list(obj, "errors_in_wrong", problems)
    _check_issue_list(obj, "correct_version_issues", problems)
    if obj.get("primary_error_category") not in ERROR_CATEGORIES:
        problems.append("primary_error_category must be one of the taxonomy categories")
    if obj.get("confidence") not in CONFIDENCE:
        problems.append(f"confidence must be one of {CONFIDENCE}")
    if not isinstance(obj.get("needs_user_review"), bool):
        problems.append("needs_user_review must be true or false")
    elif not errors and not obj["needs_user_review"]:
        problems.append("no errors recorded for the wrong version, so needs_user_review must be true")
    if not isinstance(obj.get("review_notes", ""), str):
        problems.append("review_notes must be a string")
    return problems


def validate_transcriptions(obj, expected_tokens: list[str] | None = None) -> list[str]:
    """Problems with a transcriber's file: {image token: transcription following the perception schema}."""
    if not isinstance(obj, dict):
        return ["not a JSON object keyed by image token"]
    problems: list[str] = []
    for token in expected_tokens or []:
        if token not in obj:
            problems.append(f"{token}: no transcription")
    for token, transcription in obj.items():
        if expected_tokens is not None and token not in expected_tokens:
            problems.append(f"{token}: not one of the images you were given")
        problems += [f"{token}: {p}" for p in perception.validate(transcription)]
    return problems


def validate_verification(obj, molecule_id: str) -> list[str]:
    """Problems with the verifier's file (empty list = valid)."""
    if not isinstance(obj, dict):
        return ["not a JSON object"]
    problems: list[str] = []
    if obj.get("molecule") != molecule_id:
        problems.append(f"molecule field is {obj.get('molecule')!r}, expected {molecule_id!r}")
    checks = obj.get("claim_checks")
    if not isinstance(checks, list):
        problems.append("claim_checks must be a list")
    else:
        for i, check in enumerate(checks):
            if not isinstance(check, dict) or check.get("verdict") not in CLAIM_VERDICTS:
                problems.append(f"claim_checks[{i}].verdict must be one of {CLAIM_VERDICTS}")
    _check_issue_list(obj, "missed_differences", problems)
    if not isinstance(obj.get("transcription_disputes"), list):
        problems.append("transcription_disputes must be a list")
    return problems


def verification_status(ver: dict) -> str:
    """'confirmed' only if every claim held up and the verifier found nothing missed or disputed."""
    unresolved = [c for c in ver["claim_checks"] if c["verdict"] != "confirmed"]
    return "disputed" if unresolved or ver["missed_differences"] or ver["transcription_disputes"] else "confirmed"


CSV_COLUMNS = [
    "molecule", "status", "primary_error_category", "all_error_categories", "worst_severity",
    "errors_in_wrong_drawing", "issues_in_corrected_drawing", "confidence", "needs_human_review", "review_notes", "verification",
]


def _issues_text(items: list[dict]) -> str:
    return " | ".join(f"[{i['category']}, {i['severity']}] {i['description']}" for i in items)


def export_csv(by_molecule_dir: Path, out_path: Path, expected_ids: list[str], verification_dir: Path | None = None) -> dict:
    """One row per molecule from the adjudicators' files. Missing or invalid files still get a row, so
    nothing silently drops out. Written as UTF-8 with a BOM so Excel opens it correctly."""
    import csv

    by_molecule_dir = Path(by_molecule_dir)
    verification_dir = Path(verification_dir) if verification_dir else by_molecule_dir.parent / "verification"
    rows, counts = [], {"adjudicated": 0, "pending": 0, "invalid": 0}
    for molecule in expected_ids:
        row = {c: "" for c in CSV_COLUMNS}
        row["molecule"] = molecule
        path = by_molecule_dir / f"{molecule}.json"
        if not path.exists():
            row["status"] = "pending"
            counts["pending"] += 1
            rows.append(row)
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            problems = validate_molecule(obj, molecule)
        except ValueError as exc:
            obj, problems = None, [f"invalid JSON: {exc}"]
        if problems:
            row.update(status="invalid", review_notes="; ".join(problems)[:500])
            counts["invalid"] += 1
            rows.append(row)
            continue
        ver_path = verification_dir / f"{molecule}.json"
        verification = "unverified"
        if ver_path.exists():
            try:
                ver = json.loads(ver_path.read_text(encoding="utf-8-sig"))
                verification = verification_status(ver) if not validate_verification(ver, molecule) else "invalid verification file"
            except ValueError:
                verification = "invalid verification file"
        errors = obj["errors_in_wrong"]
        order = {s: i for i, s in enumerate(SEVERITIES)}
        row.update(
            status="adjudicated",
            primary_error_category=obj["primary_error_category"],
            all_error_categories="; ".join(sorted({e["category"] for e in errors})),
            worst_severity=min((e["severity"] for e in errors), key=order.get, default=""),
            errors_in_wrong_drawing=_issues_text(errors),
            issues_in_corrected_drawing=_issues_text(obj["correct_version_issues"]) or "none",
            confidence=obj["confidence"],
            needs_human_review="yes" if obj["needs_user_review"] or verification == "disputed" else "no",
            review_notes=obj.get("review_notes", ""),
            verification=verification,
        )
        counts["adjudicated"] += 1
        rows.append(row)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return counts


def _cell(text) -> str:
    return str(text).replace("|", "/").replace("\n", " ").strip()


def _verifier_cell(ver: dict) -> str:
    if ver["status"] == "unverified":
        return "unverified"
    if ver["status"] == "confirmed":
        return "confirmed"
    unresolved = sum(1 for c in ver["claim_checks"] if c["verdict"] != "confirmed")
    return f"disputed ({unresolved} claim(s) not confirmed, {len(ver['missed_differences'])} missed, {len(ver['transcription_disputes'])} reading dispute(s))"


def merge(by_molecule_dir: Path, gold_dir: Path, expected_ids: list[str], verification_dir: Path | None = None) -> dict:
    """Validate and combine the agents' files; write transcriptions.json, errors.json, review_table.md."""
    by_molecule_dir, gold_dir = Path(by_molecule_dir), Path(gold_dir)
    verification_dir = Path(verification_dir) if verification_dir else gold_dir / "verification"
    good: dict[str, tuple[dict, dict]] = {}
    invalid: dict[str, list[str]] = {}
    missing: list[str] = []
    unverified: list[str] = []

    def load(path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8-sig")), None
        except ValueError as exc:
            return None, f"invalid JSON in {path.name}: {exc}"

    for molecule in expected_ids:
        path = by_molecule_dir / f"{molecule}.json"
        if not path.exists():
            missing.append(molecule)
            continue
        adjudicated, err = load(path)
        problems = [err] if err else validate_molecule(adjudicated, molecule)
        ver_path = verification_dir / f"{molecule}.json"
        has_verification = ver_path.exists()
        if has_verification:
            ver, err = load(ver_path)
            ver_problems = [err] if err else validate_verification(ver, molecule)
            problems += [f"verification: {p}" for p in ver_problems]
        else:
            ver = {"status": "unverified", "claim_checks": [], "missed_differences": [], "transcription_disputes": [], "notes": ""}
        if problems:
            invalid[molecule] = problems
            continue
        if has_verification:
            ver = {**ver, "status": verification_status(ver)}
        else:
            unverified.append(molecule)  # valid adjudication, but nobody has tried to refute it yet
        good[molecule] = (adjudicated, ver)

    transcriptions, errors = {}, {}
    for molecule, (obj, ver) in good.items():
        for role in ROLES:
            transcriptions[f"{molecule}/{role}"] = obj["transcriptions"][role]
        errors[molecule] = {
            "primary": obj["primary_error_category"],
            "categories": sorted({e["category"] for e in obj["errors_in_wrong"]}),
            "errors": obj["errors_in_wrong"],
            "correct_version_issues": obj["correct_version_issues"],
            "confidence": obj["confidence"],
            "needs_user_review": obj["needs_user_review"] or ver["status"] != "confirmed",
            "review_notes": obj.get("review_notes", ""),
            "verification": ver,
        }
    gold_dir.mkdir(parents=True, exist_ok=True)
    (gold_dir / "transcriptions.json").write_text(json.dumps(transcriptions, indent=2) + "\n", encoding="utf-8")
    (gold_dir / "errors.json").write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")

    rows = ["# Gold labels for review", "",
            "One row per molecule. 'Wrong version' lists what differs from the reference; "
            "'Correct version' lists anything still off in the corrected drawing. "
            "'Verifier' is an independent agent that tried to refute the row.", "",
            "| Molecule | What is wrong in the wrong version | Primary category | Still off in the correct version | Confidence | Verifier | Review? |",
            "|---|---|---|---|---|---|---|"]
    for molecule in expected_ids:
        e = errors.get(molecule)
        if not e:
            state = "missing" if molecule in missing else "invalid"
            rows.append(f"| {molecule} | _{state}_ | | | | | |")
            continue
        wrong = "; ".join(f"[{x['category']}, {x['severity']}] {x['description']}" for x in e["errors"]) or "none found"
        fixed = "; ".join(f"[{x['category']}, {x['severity']}] {x['description']}" for x in e["correct_version_issues"]) or "none"
        flag = "YES: " + (e["review_notes"] or "see verifier findings") if e["needs_user_review"] else "no"
        rows.append(f"| {molecule} | {_cell(wrong)} | {e['primary']} | {_cell(fixed)} | {e['confidence']} | {_cell(_verifier_cell(e['verification']))} | {_cell(flag)} |")
    disputed = [m for m, e in errors.items() if e["verification"]["status"] == "disputed"]
    if disputed:
        rows += ["", "## Verifier findings", ""]
        for molecule in disputed:
            ver = errors[molecule]["verification"]
            rows.append(f"### {molecule}")
            for c in ver["claim_checks"]:
                if c["verdict"] != "confirmed":
                    rows.append(f"- claim **{c['verdict']}**: {_cell(c.get('claim', ''))} ({_cell(c.get('evidence', ''))})")
            for d in ver["missed_differences"]:
                rows.append(f"- missed difference [{d['category']}, {d['severity']}]: {_cell(d['description'])}")
            for d in ver["transcription_disputes"]:
                rows.append(f"- reading dispute: {_cell(json.dumps(d))}")
            if ver.get("notes"):
                rows.append(f"- note: {_cell(ver['notes'])}")
            rows.append("")
    (gold_dir / "review_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {
        "ok": sorted(good),
        "missing": missing,
        "invalid": invalid,
        "unverified": unverified,
        "needs_review": sorted(m for m, e in errors.items() if e["needs_user_review"]),
    }
