"""pipeline_google10.py: confirming calls, the approval guard, cost tracking, output safety, the v2 CSV schema, and the removal of
the old-prompt and probe paths. No network: the model is a fake client and images are one-pixel PNGs.
"""
import base64
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import tempfile
import types
import unittest
from pathlib import Path

from chemverify import cli, decision, parse, policy
from tests.test_pairs_runner import FakeClient
from tests.test_policy_decision import CALLS, ROOT, rec

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
POLICY_ACCEPT = json.dumps({"verdict": "ACCEPT", "chemical_equivalence": True, "checklist": {a: "match" for a in policy.CHECKLIST_ASPECTS},
                            "discrepancies": [], "notation_differences": [], "unreadable_regions": [], "specific_evidence": "same",
                            "confidence": 0.9, "requires_review": False})
POLICY_REJECT = json.dumps({"verdict": "REJECT", "chemical_equivalence": False,
                            "checklist": {**{a: "match" for a in policy.CHECKLIST_ASPECTS}, "connectivity": "mismatch"},
                            "discrepancies": [{"category": "ring_size_or_fusion", "location": "central ring", "reference": "six atoms", "annotated": "five atoms"}],
                            "notation_differences": [], "unreadable_regions": [], "specific_evidence": "ring", "confidence": 0.9, "requires_review": False})
CONFIRMED = json.dumps({"claim_status": "confirmed", "image_a_observation": "six ring atoms", "image_b_observation": "five ring atoms",
                        "difference_is_chemical": True, "unreadable_regions": [], "specific_evidence": "counted", "confidence": 0.8})


def load_pipeline():
    spec = importlib.util.spec_from_file_location("pipeline_google10", ROOT / "pipeline_google10.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScriptedClient(FakeClient):
    """Answers a confirming call (its last part is the claim message) with CONFIRMED and a policy call with `policy_text`."""

    def __init__(self, policy_text=POLICY_ACCEPT, confirm_text=CONFIRMED, **kw):
        super().__init__(**kw)
        self.policy_text, self.confirm_text = policy_text, confirm_text

    def generate(self, system_prompt, parts, run_idx=0):
        self.text = self.confirm_text if "Claim to test" in str(getattr(parts[-1], "text", "")) else self.policy_text
        return super().generate(system_prompt, parts, run_idx)


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        self.pg = load_pipeline()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.pg.DATA10 = self.dir / "data"

    def mols(self, ids):
        out = []
        for i in ids:
            (self.dir / "data" / "images" / i).mkdir(parents=True)
            for role in ("ref", "annotated"):
                (self.dir / "data" / "images" / i / f"{role}.png").write_bytes(PNG)
            image = {"file": "", "sha256": hashlib.sha256(PNG).hexdigest(), "source": f"https://drive.google.com/file/d/{i}/view"}
            out.append({"id": i, "complete": True, "images": {r: {**image, "file": f"images/{i}/{r}.png"} for r in ("ref", "annotated")}})
        return out

    @staticmethod
    def policy_records(mol_id, texts, cost=0.01):
        return {f"{mol_id}__annotated": [{"task_id": f"{mol_id}__annotated", "run_idx": i, "status": "ok", "raw_text": t,
                                         "data": parse.parse_verdict_json(t).data, "usage": {"cost": cost}} for i, t in enumerate(texts)]}

    def args(self, **over):
        base = dict(workers=1, thinking_level="low", backend="openrouter", budget_usd=None, max_calls=50, only=None, dry_run=False,
                    no_confirm=False, approve_confirm_prompt=load_prompt_sha())
        return types.SimpleNamespace(**{**base, **over})


def load_prompt_sha():
    from chemverify.prompts import load_prompt
    return load_prompt(str(ROOT / "prompts" / "chem_equiv_confirm_v1.txt")).sha[:12]


class ConfirmTaskTests(PipelineTestCase):
    def test_the_claim_travels_with_the_images_and_is_part_of_the_record_key(self):
        mol = self.mols(["A"])[0]
        claim = {"category": "ring_size_or_fusion", "location": "central ring", "reference": "6 atoms", "annotated": "5 atoms"}
        task = self.pg.confirm_task(mol, claim)
        self.assertEqual(task.id, "A__confirm")
        self.assertEqual(task.meta["claim_key"], decision.claim_key("ring_size_or_fusion", "central ring"))
        parts = task.load_parts()
        self.assertEqual(len(parts), 5)  # label A, image A, label B, image B, claim message
        self.assertEqual(parts[-1].text, policy.claim_message(claim))
        other = self.pg.confirm_task(mol, {**claim, "location": "left ring"})
        self.assertNotEqual(task.input_fp, other.input_fp)  # a different claim is a different call


class EscalationTests(PipelineTestCase):
    def run_escalate(self, mols, evidence, client, **over):
        args = self.args(**over)
        made = self.pg.escalate(args, mols, self.dir / "policy", self.dir / "confirm", {}, evidence, client=client)
        confirm = self.pg.by_task(self.dir / "confirm") if (self.dir / "confirm" / "outputs.jsonl").exists() else {}
        policy_recs = evidence if evidence is not None else (self.pg.by_task(self.dir / "policy") if (self.dir / "policy" / "outputs.jsonl").exists() else {})
        return made, self.pg.compute_decisions(mols, policy_recs, confirm, {}, final=True), confirm

    def test_recorded_evidence_only_gets_confirming_calls_for_unconfirmed_rejects(self):
        mols = self.mols(["A", "B"])
        evidence = {**self.policy_records("A", [POLICY_REJECT]), **self.policy_records("B", [POLICY_ACCEPT, POLICY_ACCEPT])}
        client = ScriptedClient(result_kw={"usage": {"cost": 0.25}})
        made, decisions, confirm = self.run_escalate(mols, evidence, client)
        self.assertEqual(len(client.calls), 1)  # one confirming call for A; B is resolved; no policy call because the evidence is recorded
        self.assertEqual((made["confirm_calls"], round(made["confirm_usd"], 2), made["policy_calls"]), (1, 0.25, 0))
        a, b = decisions["A"], decisions["B"]
        self.assertEqual((a.outcome, a.rule, a.evidence_status, a.consistency_status), ("REJECT", "confirmed_reject", "confirmed", "ok"))
        self.assertEqual((b.outcome, b.rule), ("ACCEPT", "verified_equivalence"))
        record = confirm["A__confirm"][0]
        self.assertEqual(record["claim_key"], decision.claim_key("ring_size_or_fusion", "central ring"))
        meta = json.loads((self.dir / "confirm" / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["stage"], meta["decision_version"], meta["format"], meta["approved_prompt_sha"]),
                         ("confirm", policy.DECISION_VERSION, "claim_json", load_prompt_sha()))
        ledger = self.pg.cost_ledger(mols, evidence, confirm)
        self.assertEqual(ledger["per_pair"]["A"], {"policy_calls": 1, "policy_cost_usd": 0.01, "confirm_calls": 1, "confirm_cost_usd": 0.25})
        self.assertAlmostEqual(ledger["totals"]["total_cost_usd"], 0.28, places=4)

    def test_no_confirming_call_without_the_approved_prompt_hash(self):
        mols = self.mols(["A"])
        evidence = self.policy_records("A", [POLICY_REJECT])
        for over in ({"approve_confirm_prompt": None}, {"no_confirm": True}):
            client = ScriptedClient()
            _, decisions, _ = self.run_escalate(mols, evidence, client, **over)
            self.assertEqual(client.calls, [])
            self.assertEqual((decisions["A"].outcome, decisions["A"].rule), ("UNCERTAIN", "confirmation_not_run"), over)
        with self.assertRaises(SystemExit):  # a hash that does not match the file on disk is refused outright
            self.run_escalate(mols, evidence, ScriptedClient(), approve_confirm_prompt="000000000000")

    def test_a_confirming_call_that_cannot_confirm_leaves_the_pair_uncertain(self):
        mols = self.mols(["A"])
        evidence = self.policy_records("A", [POLICY_REJECT])
        refuting = ScriptedClient(confirm_text=json.dumps({"claim_status": "refuted", "image_a_observation": "six", "image_b_observation": "six",
                                                           "difference_is_chemical": False, "unreadable_regions": [], "specific_evidence": "same",
                                                           "confidence": 0.7}))
        _, decisions, _ = self.run_escalate(mols, evidence, refuting)
        d = decisions["A"]
        self.assertEqual((d.outcome, d.rule, d.evidence_status, d.requires_review, d.consistency_status), ("UNCERTAIN", "checks_disagree", "refuted", True, "ok"))

    def test_fresh_mode_makes_policy_calls_then_confirms_and_stops_when_resolved(self):
        mols = self.mols(["A", "B"])
        (self.dir / "policy").mkdir()
        record = {"key": "k1", "task_id": "A__annotated", "run_idx": 0, "status": "ok", "raw_text": POLICY_REJECT,
                  "data": parse.parse_verdict_json(POLICY_REJECT).data, "usage": {"cost": 0.02}}
        (self.dir / "policy" / "outputs.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        client = ScriptedClient(result_kw={"usage": {"cost": 0.05}})
        made, decisions, confirm = self.run_escalate(mols, None, client)
        self.assertEqual(made["policy_calls"], 2)  # B needs two clean accepts; A is a reject and needs none
        self.assertEqual(made["confirm_calls"], 1)
        self.assertEqual((decisions["A"].rule, decisions["B"].rule), ("confirmed_reject", "verified_equivalence"))
        self.assertEqual(sum(1 for t, _ in client.calls if t.startswith("Image A")), 3)  # every call labels its first image "Image A ..."

    def test_dry_run_makes_no_call_and_writes_no_run_folder(self):
        mols = self.mols(["A"])
        client = ScriptedClient()
        made, _, _ = self.run_escalate(mols, self.policy_records("A", [POLICY_REJECT]), client, dry_run=True)
        self.assertEqual((client.calls, made["confirm_calls"]), ([], 0))
        self.assertFalse((self.dir / "confirm").exists())


class GuardTests(PipelineTestCase):
    def test_existing_outputs_are_never_replaced_by_accident(self):
        path = self.dir / "out.csv"
        self.assertEqual(self.pg.new_output(str(path), False), str(path))
        path.write_text("x", encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.pg.new_output(str(path), False)
        self.assertEqual(self.pg.new_output(str(path), True), str(path))

    def test_a_recorded_run_is_never_written_into(self):
        run = self.dir / "run"
        self.pg.writable_run_dir(run, False)  # does not exist yet: fine
        run.mkdir()
        (run / "meta.json").write_text(json.dumps({"kind": "pairs", "prompt_sha": "x"}), encoding="utf-8")
        for resume in (False, True):
            with self.assertRaises(SystemExit):  # exists without --resume, or exists but was not made by this pipeline version
                self.pg.writable_run_dir(run, resume)
        (run / "meta.json").write_text(json.dumps({"decision_version": policy.DECISION_VERSION}), encoding="utf-8")
        self.pg.writable_run_dir(run, True)  # a run of this version may be resumed
        with self.assertRaises(SystemExit):
            self.pg.writable_run_dir(run, False)

    def test_dataset_and_tag_keep_outputs_apart(self):
        pg = self.pg
        try:
            pg.use_dataset("google30")
            pg.use_tag("v2")
            self.assertEqual(Path(pg.run_file("results.csv")).name, "google30_v2_results.csv")
            self.assertEqual(Path(pg.run_file("policy")).name, "google30_v2_policy")
            pg.use_tag("")
            self.assertEqual(Path(pg.run_file("results.csv")).name, "google30_results.csv")
            for bad in ("../x", "V2", "a b"):
                with self.assertRaises(SystemExit):
                    pg.use_tag(bad)
            with self.assertRaises(SystemExit):
                pg.use_dataset("Google 30")
        finally:
            pg.use_dataset("google10")
            pg.use_tag("")


class SchemaAndRemovalTests(PipelineTestCase):
    OLD = {"verdict", "verdict_label", "reason", "run_decisions", "n_runs", "accept_votes", "reject_votes", "cannot_verify_votes", "agreement",
           "stated_confidence", "conf_prompt_verdict", "perception_differences", "critical_fields_agree", "perception_differences_policy",
           "annotated_ink", "annotated_legibility", "vote_agreement"}

    def test_schema_v2(self):
        pg = self.pg
        self.assertFalse(self.OLD & set(pg.COLUMNS))
        self.assertTrue(set(pg.FINAL_COLUMNS) <= set(pg.COLUMNS))
        self.assertEqual(pg.FINAL_COLUMNS[:2], ["IDs", "final_outcome"])
        for needed in ("final_explanation", "evidence_status", "consistency_status", "confirm_calls", "policy_cost_usd", "confirm_cost_usd",
                       "api_cost_usd", "evidence_source", "decision_version", "schema_version", "accept_share"):
            self.assertIn(needed, pg.COLUMNS)

    def test_the_old_prompt_and_the_probe_are_gone_from_the_pipeline_but_not_from_the_harness(self):
        source = (ROOT / "pipeline_google10.py").read_text(encoding="utf-8")
        for gone in ("baseline_v0", "perception_probe", "PROBE_INSTRUCTION", "perception_json", "legacy_rows", "judge_legacy", "import perception"):
            self.assertNotIn(gone, source, gone)
        for name in ("run_kind", "legacy_rows", "field_diff", "cmd_summary", "cmd_csv"):
            self.assertFalse(hasattr(self.pg, name), name)
        with contextlib.redirect_stdout(io.StringIO()):  # an unknown command prints the usage text and returns 2
            self.assertEqual((self.pg.main(["probe"]), self.pg.main(["verify"]), self.pg.main(["summary"]), self.pg.main(["csv"])), (2, 2, 2, 2))
        args = cli.build_parser().parse_args(["probe", "--prompt", "p", "--name", "n"])  # the harness probe is untouched
        self.assertEqual(args.name, "n")
        self.assertTrue((ROOT / "prompts" / "perception_probe.txt").exists())
        self.assertTrue((ROOT / "archive" / "pipeline_google10_v1_before_confirmation.py").exists())

    def test_rows_explanations_and_outcomes_agree_on_recorded_calls(self):
        pg = self.pg
        keys = [k for k in CALLS if k.startswith("google30/")]
        mols = self.mols([k.split("/")[1] for k in keys])
        policy_recs = {}
        for k in keys:
            m = k.split("/")[1]
            policy_recs[f"{m}__annotated"] = [{"task_id": f"{m}__annotated", "run_idx": c["run_idx"], "status": "ok", "raw_text": c["raw_text"],
                                               "data": parse.parse_verdict_json(c["raw_text"]).data, "usage": c["usage"]} for c in CALLS[k]]
        decisions = pg.compute_decisions(mols, policy_recs, {}, {}, final=True)
        ledger = pg.cost_ledger(mols, policy_recs, {})
        rows = pg.build_rows(mols, decisions, ledger, "recorded:test")
        path = self.dir / "r.csv"
        pg.write_csv(str(path), pg.COLUMNS, rows)
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            self.assertEqual(reader.fieldnames, pg.COLUMNS)
            written = list(reader)
        self.assertEqual(len(written), len(mols))
        for row in written:
            self.assertTrue(row["final_explanation"].startswith(row["final_outcome"]), row["IDs"])
            self.assertEqual(row["consistency_status"], "ok")
            self.assertNotEqual(row["final_outcome"], "REJECT")  # none of these has a confirming call
            self.assertEqual(row["evidence_source"], "recorded:test")
            self.assertEqual((row["decision_version"], row["schema_version"]), (policy.DECISION_VERSION, policy.SCHEMA_VERSION))
            if row["final_outcome"] == "UNCERTAIN":
                self.assertEqual((row["requires_review"], bool(row["failure_categories"])), ("yes", True))
            else:
                self.assertEqual((row["requires_review"], row["failure_categories"], row["rejection_evidence"]), ("no", "", ""))
        self.assertAlmostEqual(sum(float(r["api_cost_usd"]) for r in written), ledger["totals"]["total_cost_usd"], places=3)

    def test_decide_end_to_end_offline_refuses_to_overwrite(self):
        pg = self.pg
        mols = self.mols(["A", "B"])
        pg.MANIFEST10 = self.dir / "data" / "manifest.json"
        pg.MANIFEST10.write_text(json.dumps({"version": 1, "molecules": mols}), encoding="utf-8")
        evidence = self.dir / "recorded"
        evidence.mkdir()
        lines = []
        for mol, texts in (("A", [POLICY_REJECT]), ("B", [POLICY_ACCEPT, POLICY_ACCEPT])):
            for i, t in enumerate(texts):
                lines.append(json.dumps({"key": f"{mol}{i}", "task_id": f"{mol}__annotated", "run_idx": i, "status": "ok", "raw_text": t,
                                         "data": parse.parse_verdict_json(t).data, "usage": {"cost": 0.01}}))
        (evidence / "outputs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        out = [str(self.dir / n) for n in ("results.csv", "final.csv", "decisions.json")]
        argv = ["--evidence-from", str(evidence), "--out", out[0], "--final-out", out[1], "--json-out", out[2],
                "--policy-run", str(self.dir / "p"), "--confirm-run", str(self.dir / "c")]
        self.assertEqual(pg.cmd_decide(argv), 0)
        data = json.loads(Path(out[2]).read_text(encoding="utf-8"))
        self.assertEqual((data["decision_version"], data["schema_version"], data["evidence_source"]), (policy.DECISION_VERSION, policy.SCHEMA_VERSION, "recorded:recorded"))
        self.assertEqual(data["decisions"]["A"]["rule"], "confirmation_not_run")  # no live call was made, so the reject stays unconfirmed
        self.assertEqual(data["decisions"]["B"]["outcome"], "ACCEPT")
        self.assertEqual((data["cost"]["confirm_calls"], data["cost"]["policy_calls"], data["cost"]["total_cost_usd"]), (0, 3, 0.03))
        with self.assertRaises(SystemExit):  # the outputs now exist: a second run must not replace them
            pg.cmd_decide(argv)
        self.assertFalse((self.dir / "c").exists())

    def test_confusion_metrics(self):
        c = self.pg.confusion([(True, True)] * 7 + [(False, True)] * 2)
        self.assertEqual((c["TP"], c["FP"], c["FN"], c["TN"], c["precision"], c["recall"], c["f1"]), (7, 0, 2, 0, 1.0, 0.778, 0.875))
        self.assertIsNone(c["specificity"])
        empty = self.pg.confusion([(False, False)])
        self.assertEqual((empty["precision"], empty["recall"], empty["f1"]), (None, None, None))


class CostLedgerTests(PipelineTestCase):
    def test_errors_are_not_counted_and_stages_are_kept_apart(self):
        mols = self.mols(["A"])
        policy_recs = self.policy_records("A", [POLICY_ACCEPT, POLICY_ACCEPT], cost=0.04)
        policy_recs["A__annotated"].append({"task_id": "A__annotated", "run_idx": 2, "status": "error", "usage": {"cost": 9.0}})
        confirm = {"A__confirm": [{"task_id": "A__confirm", "run_idx": 0, "status": "ok", "usage": {"cost": 0.03}},
                                  {"task_id": "A__confirm", "run_idx": 1, "status": "error"}]}
        ledger = self.pg.cost_ledger(mols, policy_recs, confirm)
        self.assertEqual(ledger["per_pair"]["A"], {"policy_calls": 2, "policy_cost_usd": 0.08, "confirm_calls": 1, "confirm_cost_usd": 0.03})
        self.assertEqual(ledger["totals"], {"policy_calls": 2, "policy_cost_usd": 0.08, "confirm_calls": 1, "confirm_cost_usd": 0.03, "total_cost_usd": 0.11})


if __name__ == "__main__":
    unittest.main()
