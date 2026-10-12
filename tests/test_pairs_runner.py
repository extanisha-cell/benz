import json
import tempfile
import time
import unittest
from pathlib import Path

from chemverify import pairs, runner, splits
from chemverify.gemini import GeminiError, GeminiResult, GenSettings, ImagePart, TextPart
from chemverify.prompts import PromptSpec

from .test_imageinfo import make_png


def fake_manifest(n=6):
    mols = []
    for i in range(n):
        images = {role: {"file": f"images/m{i}/{role}.png", "sha256": f"{role[:1]}{i:02d}".ljust(64, "0")} for role in ("ref", "wrong", "correct")}
        mols.append({"id": f"m{i}", "images": images, "complete": True})
    mols.append({"id": "partial", "images": {"ref": {"file": "images/partial/ref.png", "sha256": "x" * 64}}, "complete": False})
    return {"molecules": mols}


class PairTests(unittest.TestCase):
    def setUp(self):
        self.images = pairs.complete_images(fake_manifest())

    def test_incomplete_molecules_are_dropped(self):
        self.assertEqual(sorted(self.images), [f"m{i}" for i in range(6)])

    def test_build_all_kinds(self):
        built = pairs.build_pairs(self.images, sorted(self.images), include=("annotated", "cross", "identity"), identity_n=5)
        by_cat = {}
        for p in built:
            by_cat.setdefault(p.category, []).append(p)
        self.assertEqual({k: len(v) for k, v in by_cat.items()},
                         {"annotated_wrong": 6, "annotated_correct": 6, "cross_molecule": 6, "identity": 5})
        for p in by_cat["annotated_wrong"]:
            self.assertEqual((p.expected, p.gated, p.a, p.b), (0, True, f"images/{p.molecule}/ref.png", f"images/{p.molecule}/wrong.png"))
        for p in by_cat["annotated_correct"]:
            self.assertEqual((p.expected, p.b), (1, f"images/{p.molecule}/correct.png"))
        for p in by_cat["identity"]:
            self.assertEqual((p.expected, p.gated, p.a), (1, False, p.b))

    def test_cross_pairs_use_each_molecule_once_and_never_itself(self):
        cross = [p for p in pairs.build_pairs(self.images, sorted(self.images), include=("cross",))]
        self.assertEqual(sorted(p.molecule for p in cross), sorted(self.images))
        self.assertEqual(sorted(p.b_molecule for p in cross), sorted(self.images))
        for p in cross:
            self.assertNotEqual(p.molecule, p.b_molecule)
            self.assertEqual((p.expected, p.gated, p.b), (0, True, f"images/{p.b_molecule}/correct.png"))

    def test_cross_needs_two_molecules_and_kinds_are_checked(self):
        self.assertEqual(pairs.build_pairs(self.images, ["m0"], include=("cross",)), [])
        with self.assertRaises(ValueError):
            pairs.build_pairs(self.images, ["m0"], include=("bogus",))

    def test_deterministic(self):
        a = [p.id for p in pairs.build_pairs(self.images, sorted(self.images), include=("cross",), seed=3)]
        b = [p.id for p in pairs.build_pairs(self.images, sorted(self.images), include=("cross",), seed=3)]
        self.assertEqual(a, b)

    def test_selection_and_heldout_guard(self):
        split = {"dev": ["m0", "m1", "m2", "m3"], "heldout": ["m4", "m5"]}
        self.assertEqual(pairs.molecules_for("dev", self.images, split), ["m0", "m1", "m2", "m3"])
        self.assertEqual(len(pairs.molecules_for("all", self.images, split)), 6)
        self.assertFalse(pairs.touches_heldout(["m0", "m1"], split))
        self.assertTrue(pairs.touches_heldout(pairs.molecules_for("all", self.images, split), split))
        self.assertFalse(pairs.touches_heldout(["m4"], None))
        with self.assertRaises(ValueError):
            pairs.molecules_for("dev", self.images, None)


class SplitTests(unittest.TestCase):
    def test_plain_split_is_disjoint_and_seeded(self):
        ids = [f"m{i}" for i in range(17)]
        s = splits.make_split(ids, n_heldout=5, seed=7)
        self.assertEqual(len(s["heldout"]), 5)
        self.assertEqual(len(s["dev"]), 12)
        self.assertFalse(set(s["dev"]) & set(s["heldout"]))
        self.assertEqual(sorted(s["dev"] + s["heldout"]), sorted(ids))
        self.assertEqual(s, {**splits.make_split(ids, n_heldout=5, seed=7), "created": s["created"]})
        self.assertNotEqual(s["heldout"], splits.make_split(ids, n_heldout=5, seed=8)["heldout"])

    def test_stratified_split_spreads_categories(self):
        primary = {"m0": "a", "m1": "a", "m2": "a", "m3": "b", "m4": "b", "m5": "c"}
        s = splits.make_split(sorted(primary), n_heldout=3, seed=1, primary=primary)
        self.assertEqual(sorted(primary[m] for m in s["heldout"]), ["a", "b", "c"])

    def test_bounds(self):
        with self.assertRaises(ValueError):
            splits.make_split(["a", "b"], n_heldout=2)

    def test_frozen_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splits.json"
            first = splits.make_split(["a", "b", "c"], n_heldout=1)
            splits.save_split(first, path)
            with self.assertRaises(FileExistsError):
                splits.save_split(first, path)
            splits.save_split(first, path, force=True)
            self.assertEqual(splits.load_split(path)["heldout"], first["heldout"])
            self.assertIsNone(splits.load_split(Path(tmp) / "none.json"))

    def test_split_of(self):
        s = {"dev": ["a"], "heldout": ["b"]}
        self.assertEqual([splits.split_of(m, s) for m in "abc"], ["dev", "heldout", "unassigned"])
        self.assertEqual(splits.split_of("a", None), "unassigned")


class FakeClient:
    """Stands in for GeminiClient: reads the task id from the first text part."""

    def __init__(self, text="1\nok", fail_ids=(), result_kw=None, delay=0.0):
        self.settings = GenSettings()
        self.calls = []
        self.text = text
        self.fail_ids = set(fail_ids)
        self.result_kw = result_kw or {}
        self.delay = delay  # real calls take time; a budget can only stop calls that have not started yet

    def generate(self, system_prompt, parts, run_idx=0):
        task_id = parts[0].text
        self.calls.append((task_id, run_idx))
        if self.delay:
            time.sleep(self.delay)
        if task_id in self.fail_ids:
            raise GeminiError("HTTP 503: overloaded", status=503, retryable=True)
        return GeminiResult(text=self.text, finish_reason="STOP", **self.result_kw)


def make_task(task_id):
    meta = {"kind": "pair", "pair_id": task_id, "molecule": "m", "b_molecule": "m", "expected": 1, "category": "annotated_correct", "gated": True}
    return runner.Task(task_id, meta, lambda task_id=task_id: [TextPart(task_id)], "fp")


PROMPT = PromptSpec("p", "p.txt", "system text", "abc123")


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "run"
        self.meta = {"kind": "pairs", "prompt_sha": PROMPT.sha, "settings_fp": GenSettings().fingerprint(), "format": "number"}
        runner.init_run_dir(self.dir, self.meta, PROMPT)

    def go(self, client, tasks, runs=2, **kw):
        return runner.run_tasks(tasks, run_dir=self.dir, prompt=PROMPT, fmt="number", client=client, runs=runs,
                                workers=2, log=lambda m: None, **kw)

    def records(self):
        return runner.load_records(self.dir / "outputs.jsonl")

    def test_runs_then_resumes_without_new_calls(self):
        client = FakeClient()
        tasks = [make_task("A"), make_task("B")]
        self.go(client, tasks)
        self.assertEqual(len(client.calls), 4)
        recs = list(self.records().values())
        self.assertEqual(len(recs), 4)
        self.assertTrue(all(r["status"] == "ok" and r["decision"] == 1 and r["valid"] for r in recs))
        self.assertEqual({r["pair_id"] for r in recs}, {"A", "B"})
        self.go(client, tasks)
        self.assertEqual(len(client.calls), 4)

    def test_budget_stops_new_calls_but_keeps_finished_ones(self):
        tasks = [make_task(f"T{i}") for i in range(12)]
        client = FakeClient(result_kw={"usage": {"cost": 0.30}}, delay=0.05)
        summary = self.go(client, tasks, runs=1, budget_usd=0.50)
        self.assertTrue(summary["stopped_by_budget"])
        made = len(client.calls)
        self.assertGreaterEqual(made, 2)
        self.assertLess(made, 12)  # the rest never started
        self.assertEqual(len(self.records()), made)  # every call that did finish was saved
        self.assertEqual(summary["calls_not_started"], 12 - made)
        self.assertAlmostEqual(summary["spent_usd"], 0.30 * made)
        # the budget is crossed by one call (0.60); with 2 workers at most one other call was already in flight
        self.assertLessEqual(summary["spent_usd"], 0.50 + 0.30 * 2)
        self.assertLessEqual(made, 3)
        self.go(client, tasks, runs=1)  # without a budget the run resumes and finishes the rest
        self.assertEqual(len(self.records()), 12)

    def test_a_budget_stop_leaves_every_pair_with_the_same_number_of_runs(self):
        tasks = [make_task(f"T{i}") for i in range(6)]
        client = FakeClient(result_kw={"usage": {"cost": 0.30}}, delay=0.05)
        client.settings = GenSettings()
        summary = self.go(client, tasks, runs=4, budget_usd=0.90)
        self.assertTrue(summary["stopped_by_budget"])
        per_pair = {}
        for r in self.records().values():
            per_pair[r["task_id"]] = per_pair.get(r["task_id"], 0) + 1
        # 6 pairs x run 0 are queued before any run 1, and the stop comes after ~3-4 calls: no pair has more
        # than one run, and the runs that exist are run 0 of the first pairs rather than 4 runs of one pair
        self.assertEqual(max(per_pair.values()), 1)
        self.assertEqual({r["run_idx"] for r in self.records().values()}, {0})

    def test_no_budget_means_no_stop_and_cost_is_summed(self):
        client = FakeClient(result_kw={"usage": {"cost": 0.25}})
        summary = self.go(client, [make_task("A"), make_task("B")], runs=1)
        self.assertNotIn("stopped_by_budget", summary)
        self.assertEqual(runner.total_cost(self.records().values()), {"usd": 0.5, "calls": 2})
        self.assertEqual(runner.total_cost([{"usage": {}}, {}]), {"usd": 0.0, "calls": 0})

    def test_failed_calls_are_retried_on_the_next_invocation(self):
        tasks = [make_task("A"), make_task("B")]
        client = FakeClient(fail_ids={"A"})
        self.go(client, tasks)
        statuses = {(r["task_id"], r["status"]) for r in self.records().values()}
        self.assertEqual(statuses, {("A", "error"), ("B", "ok")})
        healthy = FakeClient()
        self.go(healthy, tasks)
        self.assertEqual(sorted(healthy.calls), [("A", 0), ("A", 1)])
        self.assertTrue(all(r["status"] == "ok" for r in self.records().values()))

    def test_no_retry_errors_flag(self):
        tasks = [make_task("A")]
        self.go(FakeClient(fail_ids={"A"}), tasks)
        healthy = FakeClient()
        self.go(healthy, tasks, retry_errors=False)
        self.assertEqual(healthy.calls, [])

    def test_malformed_output_is_kept_as_a_valid_run_with_no_decision(self):
        client = FakeClient(text="I think they match")
        self.go(client, [make_task("A")], runs=1)
        (rec,) = self.records().values()
        self.assertEqual((rec["status"], rec["decision"], rec["valid"]), ("ok", None, False))
        self.go(client, [make_task("A")], runs=1)
        self.assertEqual(len(client.calls), 1)

    def test_empty_and_blocked_are_not_ok(self):
        self.go(FakeClient(text="  "), [make_task("A")], runs=1)
        (rec,) = self.records().values()
        self.assertEqual(rec["status"], "empty")
        (self.dir / "outputs.jsonl").unlink()

        class Blocked(FakeClient):
            def generate(self, system_prompt, parts, run_idx=0):
                return GeminiResult(text="", block_reason="SAFETY")

        self.go(Blocked(), [make_task("A")], runs=1)
        (rec,) = self.records().values()
        self.assertEqual((rec["status"], rec["error"]), ("blocked", "blocked: SAFETY"))

    def test_run_mismatch_protects_against_mixing_prompts(self):
        other = PromptSpec("p", "p.txt", "different", "zzz999")
        with self.assertRaises(runner.RunMismatch):
            runner.init_run_dir(self.dir, {**self.meta, "prompt_sha": other.sha}, other)
        with self.assertRaises(runner.RunMismatch):
            runner.init_run_dir(self.dir, {**self.meta, "settings_fp": "other"}, PROMPT)
        runner.init_run_dir(self.dir, self.meta, PROMPT)  # same settings: fine

    def test_budget_limit_stops_before_any_call(self):
        client = FakeClient()
        with self.assertRaises(runner.BudgetExceeded):
            self.go(client, [make_task("A"), make_task("B")], runs=2, max_calls=3)
        self.assertEqual(client.calls, [])

    def test_torn_final_line_is_tolerated(self):
        self.go(FakeClient(), [make_task("A")], runs=1)
        with open(self.dir / "outputs.jsonl", "a", encoding="utf-8") as fh:
            fh.write('{"key": "torn", "status": "o')
        self.assertEqual(len(self.records()), 1)

    def test_prompt_text_is_saved_with_the_run(self):
        self.assertEqual((self.dir / "prompt.txt").read_text(encoding="utf-8").strip(), "system text")


class PairTaskTests(unittest.TestCase):
    def test_parts_order_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "images" / "a.png").write_bytes(make_png(3, 2))
            (root / "images" / "b.png").write_bytes(make_png(5, 4))
            (root / "images" / "bad.gif").write_bytes(b"GIF89a\x01\x00\x01\x00\x00\x00\x00;")
            p = pairs.Pair("p1", "m", "m", "images/a.png", "images/b.png", "a" * 64, "b" * 64, 1, "annotated_correct", True)
            (task,) = runner.pair_tasks([p], root, labels=True)
            kinds = [type(x).__name__ for x in task.load_parts()]
            self.assertEqual(kinds, ["TextPart", "ImagePart", "TextPart", "ImagePart"])
            (bare,) = runner.pair_tasks([p], root, labels=False)
            parts = bare.load_parts()
            self.assertEqual([type(x).__name__ for x in parts], ["ImagePart", "ImagePart"])
            self.assertEqual(parts[0].data, (root / "images" / "a.png").read_bytes())
            self.assertEqual(parts[0].mime, "image/png")
            self.assertNotEqual(task.input_fp, bare.input_fp)
            bad = pairs.Pair("p2", "m", "m", "images/a.png", "images/bad.gif", "a" * 64, "c" * 64, 0, "annotated_wrong", True)
            (bad_task,) = runner.pair_tasks([bad], root)
            with self.assertRaises(ValueError):
                bad_task.load_parts()


if __name__ == "__main__":
    unittest.main()
