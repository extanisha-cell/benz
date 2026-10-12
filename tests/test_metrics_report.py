import unittest

from chemverify import metrics, report, splits


def rec(task_id, run_idx, expected, decision, *, status="ok", category=None, molecule="m1", gated=True):
    return {
        "key": f"{task_id}-{run_idx}", "task_id": task_id, "run_idx": run_idx, "expected": expected,
        "decision": decision, "status": status, "molecule": molecule, "b_molecule": molecule, "gated": gated,
        "category": category or ("annotated_wrong" if expected == 0 else "annotated_correct"), "reason": "why",
    }


def runs(task_id, expected, decisions, **kw):
    return [rec(task_id, i, expected, d, **kw) for i, d in enumerate(decisions)]


class OutcomeTests(unittest.TestCase):
    def test_table(self):
        cases = [
            (0, 0, "true_reject"), (0, 1, "false_accept"), (0, 2, "unsure"), (0, None, "invalid"),
            (1, 1, "true_accept"), (1, 0, "false_reject"), (1, 2, "unsure"), (1, None, "invalid"),
        ]
        for expected, decision, want in cases:
            self.assertEqual(metrics.outcome(rec("p", 0, expected, decision)), want, (expected, decision))
        self.assertEqual(metrics.outcome(rec("p", 0, 0, 1, status="error")), "error")
        # a blocked or empty answer is a verdict on that image, not a transient failure
        self.assertEqual(metrics.outcome(rec("p", 0, 0, 1, status="blocked")), "blocked")
        self.assertEqual(metrics.outcome(rec("p", 0, 0, None, status="empty")), "blocked")


class SummaryTests(unittest.TestCase):
    def test_any_run_accept_marks_the_wrong_pair(self):
        r = runs("w1", 0, [0, 0, 1]) + runs("w2", 0, [0, 0, 0])
        s = metrics.compute(r)["gated"]
        self.assertEqual(s["negatives"]["n_pairs"], 2)
        self.assertEqual(s["negatives"]["false_accept_pairs"], 1)
        self.assertEqual(s["negatives"]["false_accept_ids"], ["w1"])
        self.assertEqual((s["negatives"]["false_accept_runs"]["num"], s["negatives"]["false_accept_runs"]["den"]), (1, 6))
        self.assertGreater(s["negatives"]["upper95_one_sided"], 0.0)

    def test_zero_false_accepts_gives_exact_bound(self):
        s = metrics.compute(sum((runs(f"w{i}", 0, [0, 0]) for i in range(17)), []))["gated"]
        self.assertEqual(s["negatives"]["false_accept_pairs"], 0)
        self.assertAlmostEqual(s["negatives"]["upper95_one_sided"], 1 - 0.05 ** (1 / 17), places=6)

    def test_api_errors_are_excluded_from_denominators(self):
        r = [rec("w1", 0, 0, 0), rec("w1", 1, 0, None, status="error"), rec("w2", 0, 0, None, status="error")]
        s = metrics.compute(r)["gated"]
        self.assertEqual(s["api_errors"], 2)
        self.assertEqual(s["negatives"]["n_pairs"], 1)  # w2 never produced an answer
        self.assertEqual(s["strict_accuracy_runs"]["den"], 1)

    def test_malformed_and_unsure_are_never_accepts_and_count_as_wrong(self):
        r = runs("c1", 1, [1, None, 2, 0])
        s = metrics.compute(r)["gated"]
        self.assertEqual(s["run_outcomes"]["true_accept"], 1)
        self.assertEqual((s["run_outcomes"]["invalid"], s["run_outcomes"]["unsure"], s["run_outcomes"]["false_reject"]), (1, 1, 1))
        self.assertEqual(s["strict_accuracy_runs"]["rate"], 0.25)
        self.assertEqual(s["decided_accuracy"], 0.5)
        self.assertEqual(s["coverage"], 0.5)
        self.assertEqual(s["positives"]["false_reject_pairs"], 1)

    def test_stability(self):
        r = runs("a", 1, [1, 1, 1]) + runs("b", 1, [1, 1, 0])
        st = metrics.compute(r)["gated"]["stability"]
        self.assertEqual((st["pairs"], st["unanimous_pairs"]), (2, 1))
        self.assertAlmostEqual(st["mean_agreement"], (1.0 + 2 / 3) / 2)

    def test_identity_pairs_are_not_gated_but_are_reported(self):
        r = runs("w1", 0, [0]) + runs("self", 1, [0], category="identity", gated=False)
        out = metrics.compute(r)
        self.assertEqual(out["gated"]["n_pairs"], 1)
        self.assertEqual(out["by_category"]["identity"]["positives"]["false_reject_pairs"], 1)
        self.assertEqual(out["overall"]["n_pairs"], 2)

    def test_split_and_gold_breakdowns(self):
        r = runs("w1", 0, [0], molecule="m1") + runs("w2", 0, [1], molecule="m2") + runs("w3", 0, [0], molecule="m3")
        out = metrics.compute(
            r, split_of=lambda m: "heldout" if m == "m2" else "dev",
            gold_primary={"m1": "stereochemistry", "m2": "stereochemistry", "m3": "bond_order"})
        self.assertEqual(out["by_split"]["heldout"]["negatives"]["false_accept_pairs"], 1)
        self.assertEqual(out["by_split"]["dev"]["negatives"]["false_accept_pairs"], 0)
        self.assertEqual(out["by_gold_error"]["stereochemistry"]["negatives"]["n_pairs"], 2)
        self.assertEqual(out["by_gold_error"]["bond_order"]["negatives"]["false_accept_pairs"], 0)


class HeadlineTests(unittest.TestCase):
    def test_cross_molecule_pairs_do_not_tighten_the_bound(self):
        wrong = sum((runs(f"w{i}", 0, [0, 0], molecule=f"m{i}") for i in range(12)), [])
        cross = sum((runs(f"x{i}", 0, [0, 0], molecule=f"m{i}", category="cross_molecule") for i in range(12)), [])
        out = metrics.compute(wrong + cross)
        summary, pairs, basis = metrics.headline(out)
        self.assertEqual(basis, "annotated")
        self.assertEqual(summary["negatives"]["n_pairs"], 12)
        self.assertAlmostEqual(summary["negatives"]["upper95_one_sided"], 1 - 0.05 ** (1 / 12), places=6)
        self.assertEqual(len(pairs), 12)
        self.assertEqual(out["gated"]["negatives"]["n_pairs"], 24)  # the diluted view still exists, but is not the headline

    def test_cross_molecule_false_accept_is_not_hidden(self):
        out = metrics.compute(runs("w0", 0, [0]) + runs("x0", 0, [1], category="cross_molecule"))
        self.assertEqual(metrics.headline(out)[0]["negatives"]["false_accept_pairs"], 0)
        self.assertEqual(out["by_category"]["cross_molecule"]["negatives"]["false_accept_pairs"], 1)
        text = report.render(ReportTests.META, out)
        self.assertIn("cross-molecule sanity check", text)

    def test_falls_back_to_gated_without_annotator_pairs(self):
        out = metrics.compute(runs("x0", 0, [0], category="cross_molecule"))
        self.assertEqual(metrics.headline(out)[2], "gated")


class BlockedTests(unittest.TestCase):
    def test_blocked_stays_in_denominators_and_is_never_an_accept(self):
        r = [rec("w1", 0, 0, None, status="blocked"), rec("w1", 1, 0, 0), rec("w2", 0, 0, None, status="empty")]
        s = metrics.compute(r)["gated"]
        self.assertEqual(s["api_errors"], 0)
        self.assertEqual(s["run_outcomes"]["blocked"], 2)
        self.assertEqual(s["negatives"]["n_pairs"], 2)  # w2 only ever came back empty, and is still counted
        self.assertEqual(s["negatives"]["false_accept_pairs"], 0)
        self.assertEqual((s["blocked_runs"]["num"], s["blocked_runs"]["den"]), (2, 3))
        self.assertEqual(s["coverage"], 1 / 3)
        self.assertAlmostEqual(s["strict_accuracy_runs"]["rate"], 1 / 3)

    def test_blocked_is_reported(self):
        out = metrics.compute([rec("c1", 0, 1, None, status="blocked"), rec("c1", 1, 1, 1)])
        text = report.render(ReportTests.META, out)
        self.assertIn("blocked or came back empty", text)
        self.assertIn("blocked / empty", text)


class FlaggedCorrectTests(unittest.TestCase):
    ISSUES = {"m1": [{"category": "ink_colour", "severity": "medium", "description": "one blue stroke"}]}

    def test_accepting_a_flagged_correct_drawing_is_called_out(self):
        r = (runs("m1__correct", 1, [1, 1], molecule="m1") + runs("m2__correct", 1, [1, 0], molecule="m2")
             + runs("m3__correct", 1, [0, 0], molecule="m3"))
        issues = {**self.ISSUES, "m3": [{"category": "legibility", "severity": "low", "description": "smudge"}]}
        fc = metrics.compute(r, correct_issues=issues)["flagged_correct"]
        self.assertEqual((fc["n_flagged"], fc["n_correct_pairs"]), (2, 3))
        self.assertEqual(fc["accepted_by_any"], ["m1__correct"])
        self.assertEqual(fc["rejected_by_any"], ["m3__correct"])
        self.assertEqual((fc["unflagged"]["n_pairs"], fc["unflagged"]["false_reject_pairs"]), (1, 1))  # m2 has no issue, one run rejected it

    def test_report_section(self):
        out = metrics.compute(runs("m1__correct", 1, [1, 1], molecule="m1"), correct_issues=self.ISSUES)
        text = report.render(ReportTests.META, out)
        self.assertIn("possible false accept", text)
        self.assertIn("one blue stroke", text)

    def test_absent_without_gold(self):
        out = metrics.compute(runs("m1__correct", 1, [1], molecule="m1"))
        self.assertIsNone(out["flagged_correct"])
        self.assertNotIn("still have an issue", report.render(ReportTests.META, out))


class HeldoutHidingTests(unittest.TestCase):
    SPLIT = {"dev": ["m1"], "heldout": ["m2"]}

    def test_drop_heldout_removes_both_sides_of_a_pair(self):
        r = (runs("w1", 0, [0], molecule="m1") + runs("w2", 0, [1], molecule="m2")
             + [{**rec("x", 0, 0, 0, category="cross_molecule", molecule="m1"), "b_molecule": "m2"}])
        kept, removed = splits.drop_heldout(r, self.SPLIT)
        self.assertEqual({k["task_id"] for k in kept}, {"w1"})
        self.assertEqual(removed, 2)

    def test_totals_no_longer_reveal_a_heldout_false_accept(self):
        r = runs("w1", 0, [0], molecule="m1") + runs("w2", 0, [1], molecule="m2")
        kept, removed = splits.drop_heldout(r, self.SPLIT)
        out = metrics.compute(kept)
        self.assertEqual(metrics.headline(out)[0]["negatives"]["n_pairs"], 1)
        self.assertEqual(metrics.headline(out)[0]["negatives"]["false_accept_pairs"], 0)
        text = report.render(ReportTests.META, out, heldout_excluded=removed)
        self.assertIn("1 held-out pair(s) are left out", text)

    def test_no_split_keeps_everything(self):
        r = runs("w1", 0, [0])
        self.assertEqual(splits.drop_heldout(r, None), (r, 0))


class ProductionRuleTests(unittest.TestCase):
    def test_vote_rules_per_k(self):
        # wrong: w1 accepted by 1 of 3 calls, w2 by 3 of 3, w3 by 0. correct: c1 accepted 3/3, c2 2/3, c3 0/3
        r = (runs("w1", 0, [1, 0, 0]) + runs("w2", 0, [1, 1, 1]) + runs("w3", 0, [0, 0, 0])
             + runs("c1", 1, [1, 1, 1]) + runs("c2", 1, [1, 1, 0]) + runs("c3", 1, [0, 2, None]))
        vr = metrics.compute(r)["vote_rules"]
        self.assertEqual((vr["n_calls"], vr["n_negatives"], vr["n_positives"]), (3, 3, 3))
        by_k = {row["k"]: (row["false_accept_pairs"], row["false_reject_pairs"]) for row in vr["rules"]}
        # k=1: w1, w2 accepted / c3 sent back.  k=2: only w2 / c3.  k=3: only w2 / c2 and c3.
        self.assertEqual(by_k, {1: (2, 1), 2: (1, 1), 3: (1, 2)})
        # per run: wrong accepted 1 + 3 of 9; correct not accepted: c2 once, c3 three times
        self.assertEqual(vr["single_call"], {"false_accept_runs": 4, "n_negative_runs": 9, "sent_back_runs": 4, "n_positive_runs": 9})

    def test_errors_count_as_not_one(self):
        r = [rec("c1", 0, 1, 1), rec("c1", 1, 1, None, status="error"), rec("c1", 2, 1, 1)]
        vr = metrics.compute(r)["vote_rules"]
        self.assertEqual(vr["n_calls"], 3)
        self.assertEqual({row["k"]: row["false_reject_pairs"] for row in vr["rules"]}, {1: 0, 2: 0, 3: 1})

    def test_pairs_needed(self):
        self.assertEqual(metrics.pairs_needed(0.05), 59)
        self.assertEqual(metrics.pairs_needed(0.10), 29)
        self.assertEqual(metrics.pairs_needed(0.02), 149)

    def test_report_shows_rule_table_and_sample_size_note(self):
        r = runs("w1", 0, [0, 1]) + runs("c1", 1, [1, 1])
        text = report.render(ReportTests.META, metrics.compute(r))
        self.assertIn("Production rule", text)
        self.assertIn("all 2 say 1", text)
        self.assertIn("any of 2 says 1", text)
        self.assertIn("below 5.0%: 59", text)
        self.assertIn("1 of them were accepted", text)

    def test_single_call_run_has_no_vote_rows(self):
        text = report.render(ReportTests.META, metrics.compute(runs("w1", 0, [0]) + runs("c1", 1, [1])))
        self.assertIn("1 call per drawing", text)
        self.assertNotIn("any of", text)
        self.assertNotIn("at least", text)


class SpendTests(unittest.TestCase):
    def test_cost_line(self):
        out = metrics.compute(runs("w1", 0, [0]))
        text = report.render(ReportTests.META, out, spend={"usd": 1.2345, "calls": 10})
        self.assertIn("$1.23 over 10 calls", text)
        self.assertNotIn("Cost the API reported", report.render(ReportTests.META, out, spend={"usd": 0.0, "calls": 0}))
        self.assertNotIn("Cost the API reported", report.render(ReportTests.META, out))


class ModelVersionTests(unittest.TestCase):
    def test_warns_when_versions_are_mixed(self):
        out = metrics.compute(runs("w1", 0, [0]))
        mixed = report.render(ReportTests.META, out, model_versions={"v-001": 3, "v-002": 2})
        self.assertIn("more than one model version", mixed)
        single = report.render(ReportTests.META, out, model_versions={"v-001": 5})
        self.assertIn("v-001", single)
        self.assertNotIn("more than one model version", single)

    def test_compare_shows_non_default_settings(self):
        meta = {**ReportTests.META, "settings": {"model": "m", "thinking_level": "low", "temperature": None}}
        out = metrics.compute(runs("w1", 0, [0]))
        table = report.render_compare([(meta, out), (ReportTests.META, out)])
        self.assertIn("thinking_level=low", table)
        self.assertIn("API defaults", table)


class ReportTests(unittest.TestCase):
    META = {"name": "t", "prompt_name": "p", "prompt_sha": "abc", "format": "number", "settings": {"model": "m", "temperature": None}}

    def make(self):
        r = runs("w1", 0, [0, 0]) + runs("w2", 0, [1, 0], molecule="m2") + runs("c1", 1, [1, 1]) + runs("c2", 1, [0, 0], molecule="m2")
        return metrics.compute(r, split_of=lambda m: "heldout" if m == "m2" else "dev")

    def test_render_hides_heldout_failures_by_default(self):
        out = self.make()
        split_of = lambda m: "heldout" if m == "m2" else "dev"
        hidden = report.render(self.META, out, split_of=split_of)
        self.assertIn("Gate view", hidden)
        self.assertNotIn("w2", hidden)
        self.assertNotIn("c2", hidden)
        self.assertIn("hidden", hidden)
        shown = report.render(self.META, out, split_of=split_of, show_heldout=True)
        self.assertIn("w2", shown)
        self.assertIn("c2", shown)

    def test_render_mentions_bound_and_confusion(self):
        text = report.render(self.META, self.make())
        self.assertIn("confidence", text)
        self.assertIn("should reject", text)
        self.assertIn("should accept", text)

    def test_compare_table(self):
        table = report.render_compare([(self.META, self.make())])
        self.assertIn("| t | p | m |", table)


if __name__ == "__main__":
    unittest.main()
