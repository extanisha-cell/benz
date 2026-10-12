import json
import math
import unittest

from chemverify import parse, stats


class StatsTests(unittest.TestCase):
    def test_zero_failures_bounds(self):
        self.assertAlmostEqual(stats.clopper_pearson_upper(0, 17), 1 - 0.05 ** (1 / 17), places=6)
        lo, hi = stats.clopper_pearson(0, 15)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1 - 0.025 ** (1 / 15), places=6)

    def test_known_exact_interval(self):
        lo, hi = stats.clopper_pearson(5, 10)
        self.assertAlmostEqual(lo, 0.1871, places=3)
        self.assertAlmostEqual(hi, 0.8129, places=3)

    def test_edges(self):
        self.assertEqual(stats.clopper_pearson(0, 0), (0.0, 1.0))
        self.assertEqual(stats.clopper_pearson(10, 10)[1], 1.0)
        self.assertEqual(stats.clopper_pearson_upper(10, 10), 1.0)

    def test_wilson(self):
        lo, hi = stats.wilson(27, 30)
        self.assertAlmostEqual(lo, 0.744, places=3)
        self.assertAlmostEqual(hi, 0.965, places=3)

    def test_sample_size_rule(self):
        self.assertEqual(stats.n_for_zero_failures(0.05), 59)

    def test_bootstrap_is_seeded_and_degenerate_when_constant(self):
        clusters = [(1, 2)] * 6
        self.assertEqual(stats.ratio_bootstrap(clusters, 200, 1), (0.5, 0.5))
        mixed = [(0, 2), (2, 2), (1, 2), (2, 2)]
        self.assertEqual(stats.ratio_bootstrap(mixed, 300, 3), stats.ratio_bootstrap(mixed, 300, 3))
        self.assertEqual(stats.ratio_bootstrap([(0, 0)], 10, 0), (None, None))

    def test_binom_cdf_sums_to_one(self):
        self.assertTrue(math.isclose(stats._binom_cdf(9, 9, 0.3), 1.0))
        self.assertTrue(math.isclose(stats._binom_cdf(0, 3, 0.5), 0.125))


class ParseNumberTests(unittest.TestCase):
    def test_valid(self):
        p = parse.parse_number("1\nSame compound; all labels match.")
        self.assertEqual((p.decision, p.valid, p.problems), (1, True, []))
        self.assertEqual(p.reason, "Same compound; all labels match.")
        self.assertEqual(parse.parse_number("  0 \n\nMissing methyl on C3.").decision, 0)
        self.assertEqual(parse.parse_number("2").decision, 2)

    def test_strict_rejections_keep_a_lenient_hint(self):
        for text, lenient in (("```\n1\n```", None), ("1.", 1), ("Verdict: 1", 1), ("The answer is 1", 1), ("", None)):
            p = parse.parse_number(text)
            self.assertIsNone(p.decision, text)
            self.assertFalse(p.valid, text)
            self.assertTrue(p.problems, text)
        self.assertEqual(parse.parse_number("1.").lenient_decision, 1)
        self.assertEqual(parse.parse_number("Verdict: 0").lenient_decision, 0)

    def test_digits_other_than_012_are_invalid(self):
        for text in ("3", "01", "10", "-1", "1 0"):
            self.assertFalse(parse.parse_number(text).valid, text)

    def test_extra_lines_are_flagged_but_valid(self):
        p = parse.parse_number("1\nline two\nline three")
        self.assertTrue(p.valid)
        self.assertTrue(p.problems)


class ParseJsonTests(unittest.TestCase):
    def test_auditor_json_with_fence(self):
        text = '```json\n{"chemical_equivalence":"MISMATCH","recommended_decision":"REJECT","justification":"ring differs"}\n```'
        p = parse.parse_auditor_json(text)
        self.assertEqual((p.decision, p.valid, p.reason, p.lenient_decision), (0, True, "ring differs", 0))

    def test_auditor_json_mapping_and_failures(self):
        self.assertEqual(parse.parse_auditor_json('{"recommended_decision":"accept"}').decision, 1)
        self.assertEqual(parse.parse_auditor_json('{"recommended_decision":"REVIEW"}').decision, 2)
        self.assertFalse(parse.parse_auditor_json('{"recommended_decision":"MAYBE"}').valid)
        self.assertFalse(parse.parse_auditor_json("no json here").valid)

    def test_verdict_json_mapping(self):
        accept = '{"verdict":"ACCEPT","chemical_equivalence":true,"specific_evidence":"same","confidence":0.9,"requires_review":false}'
        p = parse.parse_verdict_json(accept)
        self.assertEqual((p.decision, p.valid, p.reason, p.lenient_decision), (1, True, "same", 1))
        self.assertEqual(p.data["confidence"], 0.9)
        reject = '```json\n{"Verdict":"reject","chemical_equivalence":false,"specific_evidence":"ring size differs"}\n```'
        self.assertEqual(parse.parse_verdict_json(reject).decision, 0)
        self.assertEqual(parse.parse_verdict_json('{"verdict":"UNCERTAIN","chemical_equivalence":null}').decision, 2)

    def test_verdict_json_is_fail_closed(self):
        for text in ('{"verdict":"ACCEPT","chemical_equivalence":false}',   # contradicts itself
                     '{"verdict":"ACCEPT","chemical_equivalence":null}',
                     '{"verdict":"ACCEPT"}',                                 # no equivalence given
                     '{"verdict":"MAYBE"}',
                     "Verdict: ACCEPT ... later, Verdict: REJECT",           # two different verdicts
                     "no verdict here", ""):
            p = parse.parse_verdict_json(text)
            self.assertFalse(p.valid, text)
            self.assertIsNone(p.decision, text)

    def test_verdict_json_plain_text_fallback(self):
        p = parse.parse_verdict_json("**Verdict:** REJECT\nThe ring is fused differently.")
        self.assertEqual((p.decision, p.valid), (0, True))
        self.assertEqual(parse.parse_verdict_json("- verdict: UNCERTAIN").decision, 2)
        self.assertEqual(parse.parse_verdict_json("verdict: ACCEPT").decision, 1)  # no JSON: nothing to contradict it

    def test_claim_json_statuses(self):
        base = {"claim_status": "confirmed", "image_a_observation": "six ring atoms", "image_b_observation": "five ring atoms",
                "difference_is_chemical": True, "specific_evidence": "counted"}
        p = parse.parse_claim_json(json.dumps(base))
        self.assertEqual((p.decision, p.valid, p.reason, p.data["claim_status"]), (1, True, "counted", "confirmed"))
        self.assertEqual(parse.parse_claim_json(json.dumps({**base, "claim_status": "REFUTED", "difference_is_chemical": False})).decision, 0)
        self.assertEqual(parse.parse_claim_json(json.dumps({**base, "claim_status": "inconclusive", "difference_is_chemical": None})).decision, 2)
        self.assertTrue(parse.parse_claim_json("```json\n" + json.dumps({"Claim_Status": "refuted"}) + "\n```").valid)

    def test_claim_json_is_fail_closed_on_a_confirmation_it_cannot_support(self):
        base = {"claim_status": "confirmed", "image_a_observation": "six ring atoms", "image_b_observation": "five ring atoms", "difference_is_chemical": True}
        for over in ({"image_b_observation": "Six ring  atoms."}, {"image_a_observation": ""}, {"image_b_observation": None},
                     {"difference_is_chemical": False}, {"difference_is_chemical": None}, {"claim_status": "maybe"}):
            p = parse.parse_claim_json(json.dumps({**base, **over}))
            self.assertEqual((p.valid, p.decision), (False, None), over)
            self.assertTrue(p.problems, over)
        self.assertFalse(parse.parse_claim_json("no json here").valid)
        self.assertEqual(parse.observation_key("Six  ring-atoms."), parse.observation_key("six ring atoms"))

    def test_json_inside_prose_and_nested(self):
        obj, err = parse.extract_json_object('Here you go: {"a": {"b": 1}} thanks')
        self.assertEqual(obj, {"a": {"b": 1}})
        self.assertIsNone(err)

    def test_perception_json(self):
        p = parse.parse_perception_json('{"atom_labels": ["N"]}')
        self.assertTrue(p.valid)
        self.assertEqual(p.data["atom_labels"], ["N"])
        self.assertFalse(parse.parse_perception_json("nope").valid)


if __name__ == "__main__":
    unittest.main()
