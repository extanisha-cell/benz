"""Policy files, per-call judgement, confirming-call judgement, the deterministic decision rules, the explanation and the
consistency gate, and the behaviour of the rules on recorded Gemini calls.

No network and no images. Recorded calls come from tests/fixtures/google_policy_calls.json (policy-prompt responses only).
Tests on recorded calls assert how the RULES behave; none of them states that a molecule is chemically right or wrong.
"""
import hashlib
import itertools
import json
import random
import tempfile
import unittest
from pathlib import Path

from chemverify import decision, graphcheck, parse, policy
from chemverify.decision import ACCEPT, REJECT, UNCERTAIN, Confirmation, Judgement
from chemverify.graphcheck import GraphResult
from chemverify.prompts import load_prompt

ROOT = Path(__file__).resolve().parents[1]
CALLS = json.loads((ROOT / "tests" / "fixtures" / "google_policy_calls.json").read_text(encoding="utf-8"))["calls"]

MATCH_ALL = {a: "match" for a in policy.CHECKLIST_ASPECTS}
CLEAN = {"verdict": "ACCEPT", "chemical_equivalence": True, "checklist": MATCH_ALL, "discrepancies": [], "notation_differences": [],
         "unreadable_regions": [], "mismatch_type": None, "specific_evidence": "same molecule", "confidence": 0.9, "requires_review": False}
RING = {"category": "ring_size_or_fusion", "location": "right-hand phenyl ring", "reference": "six-membered ring", "annotated": "five-membered ring"}
CIRCLE = {"category": "aromatic_depiction", "location": "both benzene rings", "reference": "alternating double bonds", "annotated": "aromatic circle"}
SUPPORTED = {"category": "connectivity", "location": "x", "reference": "a", "annotated": "b", "detail": ""}
KEY = decision.claim_key("connectivity", "x")


def rec(raw_text: str, status: str = "ok") -> dict:
    """A run record built the way runner.run_tasks builds it for the verdict_json format."""
    p = parse.parse_verdict_json(raw_text)
    return {"status": status, "raw_text": raw_text, "data": p.data, "valid": p.valid, "decision": p.decision}


def srec(**fields) -> dict:
    return rec(json.dumps({**CLEAN, **fields}))


def accept(**over) -> dict:
    return srec(**over)


def reject(discrepancies, **over) -> dict:
    base = {"verdict": "REJECT", "chemical_equivalence": False, "checklist": {**MATCH_ALL, "connectivity": "mismatch"}, "discrepancies": discrepancies}
    return srec(**{**base, **over})


def J(vote, **kw) -> Judgement:
    return Judgement(vote, **kw)


def VER(n):
    return [J(ACCEPT, verified=True) for _ in range(n)]


def REJ(n):
    return [J(REJECT, evidence=[dict(SUPPORTED)]) for _ in range(n)]


def CONF(status, key=KEY, **kw) -> Confirmation:
    return Confirmation(status, key, {"reference": "ref obs", "annotated": "ann obs"}, **kw)


def crec(claim_key=KEY, **fields) -> dict:
    """A confirming-call record built the way runner.run_tasks builds it for the claim_json format."""
    base = {"claim_status": "confirmed", "image_a_observation": "six ring atoms", "image_b_observation": "five ring atoms",
            "difference_is_chemical": True, "unreadable_regions": [], "specific_evidence": "counted", "confidence": 0.8}
    text = json.dumps({**base, **fields})
    p = parse.parse_claim_json(text)
    return {"status": "ok", "raw_text": text, "data": p.data, "valid": p.valid, "decision": p.decision, "claim_key": claim_key, "run_idx": 0}


class PolicyTests(unittest.TestCase):
    def test_prompt_files_are_generated_from_the_policy(self):
        for name, render in (("chem_equiv_policy_v1.txt", policy.system_prompt), ("chem_equiv_confirm_v1.txt", policy.confirmation_prompt)):
            self.assertEqual((ROOT / "prompts" / name).read_text(encoding="utf-8-sig"), render(), name)

    def test_policy_prompt_text_is_frozen(self):
        """Changing this prompt changes what Gemini sees and breaks the link to every recorded run. Bump POLICY_VERSION on purpose."""
        self.assertEqual(load_prompt(str(ROOT / "prompts" / "chem_equiv_policy_v1.txt")).sha[:12], "c610ced5ea8b")
        self.assertEqual(policy.POLICY_VERSION, "chem-equiv-1")

    def test_versions_are_separate_things(self):
        self.assertNotIn(policy.DECISION_VERSION, policy.system_prompt())  # the rules version is not part of any prompt
        self.assertNotIn(policy.DECISION_VERSION, policy.confirmation_prompt())

    def test_prompt_states_the_policy_rules_and_every_category(self):
        for text in (policy.system_prompt(), policy.confirmation_prompt()):
            for name in list(policy.CHEMICAL_CATEGORIES) + list(policy.COSMETIC_CATEGORIES):
                self.assertIn(name, text)
            for needle in ("aromatic circles and valid Kekule", "Reject only when there is an identifiable chemical mismatch", policy.POLICY_VERSION):
                self.assertIn(needle, text)

    def test_confirmation_prompt_tests_a_claim_instead_of_judging_the_pair(self):
        text = policy.confirmation_prompt()
        for needle in ("hypothesis", "Do not dismiss the claim", "never confirmed", "from Image A alone", "from Image B alone",
                       "claim_status", "difference_is_chemical", "inconclusive"):
            self.assertIn(needle, text)
        self.assertNotIn('"verdict"', text)  # it does not ask for an ACCEPT / REJECT verdict

    def test_claim_message_carries_the_whole_claim(self):
        message = policy.claim_message({"category": "ring_size_or_fusion", "location": "central ring", "reference": "6 atoms", "annotated": "5 atoms"})
        for part in ("ring_size_or_fusion", "central ring", "6 atoms", "5 atoms"):
            self.assertIn(part, message)

    def test_the_old_draft_is_archived_unchanged_and_nothing_reads_it(self):
        archived = ROOT / "prompts" / "archive" / "qc_equivalence_v1.txt"
        self.assertEqual(hashlib.sha256(archived.read_bytes()).hexdigest()[:12], "f9fcd65b0a98")
        self.assertFalse((ROOT / "prompts" / "qc_equivalence_v1.txt").exists())
        here = Path(__file__).resolve()
        readers = [str(p) for p in ROOT.rglob("*.py") if p.resolve() != here and "qc_equivalence" in p.read_text(encoding="utf-8")]
        self.assertEqual(readers, [])

    def test_readme_names_the_authoritative_policy_source(self):
        text = (ROOT / "prompts" / "README.md").read_text(encoding="utf-8")
        for needle in ("src/chemverify/policy.py", "chem_equiv_policy_v1.txt", "chem_equiv_confirm_v1.txt", "archive/"):
            self.assertIn(needle, text)

    def test_categories_do_not_overlap(self):
        self.assertFalse(set(policy.CHEMICAL_CATEGORIES) & set(policy.COSMETIC_CATEGORIES))

    def test_normalise_category(self):
        self.assertEqual(policy.normalise_category("Stereo"), "stereochemistry")
        self.assertEqual(policy.normalise_category("Aromatic ring depiction"), "aromatic_depiction")
        self.assertEqual(policy.normalise_category("ring-size"), "ring_size_or_fusion")
        self.assertEqual(policy.normalise_category("something odd"), "")

    def test_canonical_label_folds_equivalent_spellings(self):
        for group in (("Me", "CH3", "H3C", "CH₃"), ("OEt", "OC2H5", "OCH2CH3", "EtO"), ("CO2Me", "COOCH3", "MeOOC"),
                      ("CF3", "F3C", "C(F)3"), ("NHMe", "NHCH3", "CH3NH"), ("COOH", "CO2H", "HO2C")):
            self.assertEqual(len({policy.canonical_label(x) for x in group}), 1, group)
        self.assertNotEqual(policy.canonical_label("Ph"), policy.canonical_label("PH"))
        self.assertNotEqual(policy.canonical_label("CN"), policy.canonical_label("NC"))
        self.assertNotEqual(policy.canonical_label("Me"), policy.canonical_label("Et"))

    def test_classify_text(self):
        self.assertEqual(policy.classify_text("Replaces the explicit alternating double bonds with aromatic circles.")[0], {"aromatic_depiction"})
        notation, structural = policy.classify_text("Expanded the abbreviations NHMe to NHCH3 and CF3 to a carbon bonded to three fluorine atoms.")
        self.assertEqual((notation, structural), ({"abbreviation_expansion"}, False))
        self.assertTrue(policy.classify_text("The right phenyl ring is a pentagon with chlorine ortho to NH, not a hexagon with meta substitution.")[1])
        self.assertNotIn("abbreviation_expansion", policy.classify_text("The ring is expanded from five to six members.")[0])


class JudgeStructuredTests(unittest.TestCase):
    def test_clean_accept_is_verified(self):
        j = decision.judge_structured(accept())
        self.assertEqual((j.vote, j.verified, j.flags, j.confidence), (ACCEPT, True, [], 0.9))

    def test_accept_that_contradicts_itself_is_invalid(self):
        for over, flag in (({"chemical_equivalence": None}, "accept_equivalence_conflict"),
                           ({"chemical_equivalence": False}, "accept_equivalence_conflict"),
                           ({"discrepancies": [RING]}, "accept_with_discrepancy"),
                           ({"checklist": {**MATCH_ALL, "bonds": "mismatch"}}, "accept_with_discrepancy")):
            j = decision.judge_structured(accept(**over))
            self.assertEqual(j.vote, "INVALID", over)
            self.assertIn(flag, j.flags)

    def test_accept_with_unreadable_part_or_review_request_is_uncertain(self):
        self.assertEqual(decision.judge_structured(accept(unreadable_regions=["upper left"])).vote, UNCERTAIN)
        self.assertEqual(decision.judge_structured(accept(checklist={**MATCH_ALL, "stereochemistry": "unreadable"})).vote, UNCERTAIN)
        j = decision.judge_structured(accept(requires_review=True))
        self.assertEqual((j.vote, j.flags), (UNCERTAIN, ["accept_requires_review"]))

    def test_accept_without_a_complete_checklist_is_not_verified(self):
        j = decision.judge_structured(accept(checklist={"atoms": "match"}))
        self.assertEqual((j.vote, j.verified), (ACCEPT, False))
        ok = decision.judge_structured(accept(checklist={**MATCH_ALL, "charges": "not_present", "stereochemistry": "not_present"}))
        self.assertTrue(ok.verified)
        self.assertFalse(decision.judge_structured(accept(checklist={**MATCH_ALL, "atoms": "not_present"})).verified)

    def test_supported_reject_needs_a_chemical_category_and_a_location(self):
        j = decision.judge_structured(reject([RING]))
        self.assertEqual(j.vote, REJECT)
        self.assertEqual([e["category"] for e in j.evidence], ["ring_size_or_fusion"])
        no_location = decision.judge_structured(reject([{**RING, "location": ""}]))
        self.assertEqual((no_location.vote, "reject_missing_location" in no_location.flags), ("EXCLUDED", True))
        nothing = decision.judge_structured(reject([]))
        self.assertEqual((nothing.vote, "reject_without_evidence" in nothing.flags), ("EXCLUDED", True))

    def test_notation_only_reject_is_ignored_and_recorded(self):
        j = decision.judge_structured(reject([CIRCLE]))
        self.assertEqual((j.vote, j.notation), ("EXCLUDED", ["aromatic_depiction"]))
        self.assertIn("reject_notation_only", j.flags)

    def test_a_model_that_mislabels_notation_as_chemistry_is_overruled(self):
        mislabelled = {"category": "missing_or_extra_atom_or_group", "location": "top right", "reference": "Me", "annotated": "CH3"}
        j = decision.judge_structured(reject([mislabelled]))
        self.assertEqual((j.vote, j.notation), ("EXCLUDED", ["label_format"]))
        genuine = {"category": "missing_or_extra_atom_or_group", "location": "top right", "reference": "Me", "annotated": "Et"}
        self.assertEqual(decision.judge_structured(reject([genuine])).vote, REJECT)

    def test_reject_that_contradicts_its_own_equivalence_is_uncertain(self):
        j = decision.judge_structured(reject([RING], chemical_equivalence=True))
        self.assertEqual((j.vote, j.flags), (UNCERTAIN, ["reject_contradicts_equivalence"]))
        self.assertEqual(decision.judge_structured(srec(verdict="REJECT", chemical_equivalence=False, discrepancies=[RING])).vote, UNCERTAIN)

    def test_uncertain_and_reject_with_unreadable_regions(self):
        j = decision.judge_structured(srec(verdict="UNCERTAIN", chemical_equivalence=None, checklist={**MATCH_ALL, "bonds": "unreadable"},
                                           unreadable_regions=["bottom"]))
        self.assertEqual((j.vote, j.flags), (UNCERTAIN, []))
        self.assertEqual(decision.judge_structured(reject([], unreadable_regions=["left"])).vote, UNCERTAIN)

    def test_bad_calls_are_invalid_never_accepts(self):
        for r in ({"status": "blocked"}, {"status": "empty"}, rec("no json at all"), rec('{"verdict": "MAYBE"}'), rec('{"x": 1}')):
            j = decision.judge_structured(r)
            self.assertEqual((j.vote, j.verified), ("INVALID", False))

    def test_json_in_a_code_fence_is_read_from_the_raw_text(self):
        text = "```json\n" + json.dumps({**CLEAN, "Verdict": "ACCEPT"}) + "\n```"
        self.assertEqual(decision.judge_structured({"status": "ok", "raw_text": text, "data": None}).vote, ACCEPT)

    def test_confidence_is_read_as_a_number_only(self):
        for given, expected in ((0.9, 0.9), (90, 0.9), ("85%", 0.85), ("0.7", 0.7), ("high", None), (None, None), (140, None), (True, None)):
            self.assertEqual(decision.judge_structured(accept(confidence=given)).confidence, expected, given)


class JudgeConfirmationTests(unittest.TestCase):
    def test_a_clean_confirmation(self):
        c = decision.judge_confirmation(crec())
        self.assertEqual((c.status, c.claim_key, c.observations, c.confidence), ("confirmed", KEY, {"reference": "six ring atoms", "annotated": "five ring atoms"}, 0.8))

    def test_confirmed_needs_two_different_observations_and_a_chemical_difference(self):
        for over in ({"image_a_observation": "six atoms", "image_b_observation": "Six  atoms."},  # same after folding case, spacing, punctuation
                     {"image_a_observation": ""}, {"image_b_observation": "  "}, {"difference_is_chemical": False}, {"difference_is_chemical": None}):
            c = decision.judge_confirmation(crec(**over))
            self.assertEqual(c.status, "invalid", over)
            self.assertIn("confirmed_claim_invalid", c.flags)

    def test_confirmed_with_an_unreadable_region_is_only_inconclusive(self):
        c = decision.judge_confirmation(crec(unreadable_regions=["the spiro ring"]))
        self.assertEqual((c.status, c.flags), ("inconclusive", ["confirmed_with_unreadable_region"]))

    def test_refuted_and_inconclusive_pass_through(self):
        self.assertEqual(decision.judge_confirmation(crec(claim_status="refuted", difference_is_chemical=False)).status, "refuted")
        self.assertEqual(decision.judge_confirmation(crec(claim_status="inconclusive", difference_is_chemical=None)).status, "inconclusive")

    def test_bad_calls_are_invalid(self):
        for r in ({"status": "blocked", "claim_key": KEY}, {"status": "ok", "raw_text": "nothing", "data": None, "claim_key": KEY},
                  crec(claim_status="maybe"), {**crec(), "data": {"claim_status": "confirmed"}}):
            self.assertEqual(decision.judge_confirmation(r).status, "invalid")

    def test_the_decision_layer_does_not_trust_the_parsers_verdict(self):
        record = crec(image_b_observation="six ring atoms")  # identical observations: the parser already refused it
        self.assertIsNone(record["decision"])
        self.assertEqual(decision.judge_confirmation({**record, "data": None}).status, "invalid")  # and so does the re-derivation from raw_text


class DecideTests(unittest.TestCase):
    def test_accept_needs_the_quorum_of_verified_calls(self):
        one = decision.decide(VER(1))
        self.assertEqual((one.outcome, one.rule, one.needs_escalation), (UNCERTAIN, "insufficient_evidence", True))
        two = decision.decide(VER(2))
        self.assertEqual((two.outcome, two.rule, two.needs_escalation, two.requires_review, two.failure_categories, two.evidence_status),
                         (ACCEPT, "verified_equivalence", False, False, [], "none"))
        self.assertEqual(decision.decide([J(ACCEPT)] * 3).outcome, UNCERTAIN)  # accepts without a complete checklist

    def test_notation_only_rejects_do_not_block_an_accept(self):
        self.assertEqual(decision.decide(VER(2) + [J("EXCLUDED")] * 2).rule, "verified_equivalence")

    def test_a_single_reject_waits_for_its_confirming_call(self):
        d = decision.decide(REJ(1))
        self.assertEqual((d.outcome, d.rule, d.needs_confirmation, d.needs_escalation, d.evidence_status, d.requires_review),
                         (UNCERTAIN, "reject_pending_confirmation", True, False, "asserted", True))
        off = decision.decide(REJ(1), confirmation_enabled=False)
        self.assertEqual((off.rule, off.needs_confirmation, off.failure_categories), ("confirmation_not_run", False, ["confirmation_not_run"]))

    def test_repeated_rejects_are_not_corroboration(self):
        d = decision.decide(REJ(3))  # three calls of one model on one image
        self.assertEqual((d.outcome, d.rule, d.needs_confirmation, d.claim["votes"]), (UNCERTAIN, "reject_pending_confirmation", True, 3))

    def test_what_the_confirming_call_says_decides(self):
        confirmed = decision.decide(REJ(1), CONF("confirmed"))
        self.assertEqual((confirmed.outcome, confirmed.rule, confirmed.evidence_status, confirmed.failure_categories, confirmed.needs_confirmation),
                         (REJECT, "confirmed_reject", "confirmed", ["connectivity"], False))
        self.assertEqual(confirmed.confirmation["status"], "confirmed")
        for status, rule, evidence_status, cause in (("refuted", "checks_disagree", "refuted", "checks_disagree"),
                                                     ("inconclusive", "reject_not_confirmed", "inconclusive", "reject_not_confirmed"),
                                                     ("invalid", "reject_not_confirmed", "inconclusive", "reject_not_confirmed")):
            d = decision.decide(REJ(1), CONF(status))
            self.assertEqual((d.outcome, d.rule, d.evidence_status, d.failure_categories, d.requires_review), (UNCERTAIN, rule, evidence_status, [cause], True), status)

    def test_a_confirmation_of_a_different_claim_is_ignored(self):
        d = decision.decide(REJ(1), CONF("confirmed", key="connectivity|elsewhere"))
        self.assertEqual((d.outcome, d.rule, d.needs_confirmation), (UNCERTAIN, "reject_pending_confirmation", True))
        self.assertIn("stale_confirmation_ignored", d.flags)

    def test_a_verified_accept_keeps_any_reject_unresolved_even_if_it_was_confirmed(self):
        for js in (VER(1) + REJ(1), VER(1) + REJ(2), VER(2) + REJ(1)):
            d = decision.decide(js, CONF("confirmed"))
            self.assertEqual((d.outcome, d.rule), (UNCERTAIN, "contested_reject"), len(js))
            self.assertEqual(d.needs_escalation, len(js) < policy.MAX_STRUCTURED_CALLS)
        self.assertIn("escalation_exhausted", decision.decide(VER(1) + REJ(2)).failure_categories)

    def test_graph_check(self):
        match, mismatch = GraphResult("match"), GraphResult("mismatch", "different molecular graph", ("other_chemical",))
        d = decision.decide(REJ(1), None, mismatch)  # an independent structural check needs no confirming call
        self.assertEqual((d.outcome, d.rule, d.evidence_status, d.needs_confirmation), (REJECT, "graph_confirmed_reject", "graph_confirmed", False))
        self.assertEqual(d.evidence[-1]["category"], "other_chemical")
        both = decision.decide(VER(2), None, mismatch)
        self.assertEqual((both.outcome, both.rule, both.failure_categories), (UNCERTAIN, "graph_vlm_conflict", ["graph_conflict"]))
        self.assertTrue(decision.decide([], None, mismatch).needs_escalation)
        self.assertEqual(decision.decide(VER(1), None, match).outcome, ACCEPT)  # the graph supplies the second vote
        self.assertEqual(decision.decide([], None, match).outcome, UNCERTAIN)  # a graph match alone never accepts
        self.assertEqual(decision.decide(REJ(1), CONF("confirmed"), match).rule, "contested_reject")  # graph says equal, a call says not
        for ignored in (GraphResult("unavailable", "rdkit is not installed"), graphcheck.NOT_PROVIDED, GraphResult("unparseable")):
            self.assertEqual(decision.decide(VER(2), None, ignored).outcome, ACCEPT)

    def test_unreadable_structures_are_uncertain(self):
        d = decision.decide([J(UNCERTAIN)])
        self.assertEqual((d.outcome, d.rule, d.needs_escalation, d.requires_review), (UNCERTAIN, "unreadable_or_uncertain", True, True))
        spent = decision.decide([J(UNCERTAIN)] + VER(2))
        self.assertEqual((spent.needs_escalation, spent.requires_review), (False, True))
        self.assertTrue({"escalation_exhausted", "unreadable_region"} <= set(spent.failure_categories))

    def test_malformed_calls_are_reported(self):
        d = decision.decide([J("INVALID", flags=["unparseable_output"])] + VER(1))
        self.assertEqual(d.outcome, UNCERTAIN)
        self.assertIn("malformed_output", d.failure_categories)

    def test_agreement_and_confidence_are_separate_numbers(self):
        d = decision.decide([J(ACCEPT, verified=True, confidence=0.9), J(ACCEPT, verified=True, confidence=0.7), J("EXCLUDED")])
        self.assertEqual((d.outcome, d.accept_share, d.model_confidence), (ACCEPT, 0.67, 0.8))
        self.assertEqual(decision.votes_text(d.votes), "ACCEPT:2;REJECT:0;UNCERTAIN:0;EXCLUDED:1;INVALID:0")
        self.assertIsNone(decision.decide([J(UNCERTAIN)]).model_confidence)

    def test_model_confidence_never_changes_an_outcome(self):
        compositions = (VER(2), REJ(1), VER(1) + REJ(1), [J(UNCERTAIN)] + VER(1), REJ(2) + VER(1))
        for base in compositions:
            expected = decision.decide(base, CONF("confirmed"))
            for seed in range(10):
                rng = random.Random(seed)
                varied = [Judgement(j.vote, j.verified, j.evidence, j.notation, j.flags, rng.choice([None, 0.0, 0.5, 1.0])) for j in base]
                d = decision.decide(varied, CONF("confirmed"))
                self.assertEqual((d.outcome, d.rule), (expected.outcome, expected.rule))

    def test_decision_does_not_depend_on_call_order(self):
        base = VER(2) + [J("EXCLUDED")] + [J(UNCERTAIN)] * 0
        self.assertEqual(len({(d.outcome, d.rule) for d in (decision.decide(random.Random(s).sample(base, len(base))) for s in range(20))}), 1)
        base = REJ(1) + VER(1)
        self.assertEqual(len({(d.outcome, d.rule) for d in (decision.decide(random.Random(s).sample(base, len(base)), CONF("confirmed")) for s in range(20))}), 1)

    def test_exhaustive_invariants(self):
        """Over every small mix of votes, graph result and confirming call: a REJECT is always confirmed by something other than
        a repeated policy call, an ACCEPT always has its quorum, and uncertainty is never resolved silently."""
        confirmations = (None, CONF("confirmed"), CONF("refuted"), CONF("inconclusive"), CONF("invalid"), CONF("confirmed", key="connectivity|elsewhere"))
        grid = itertools.product(range(4), range(4), range(3), range(3), ("", "match", "mismatch"), range(len(confirmations)))
        checked = 0
        for verified, supported, uncertain, excluded, graph, c in grid:
            js = VER(verified) + REJ(supported) + [J(UNCERTAIN)] * uncertain + [J("EXCLUDED")] * excluded
            confirmation = confirmations[c]
            g = GraphResult(graph) if graph else graphcheck.NOT_PROVIDED
            d = decision.decide(js, confirmation, g)
            label = (verified, supported, uncertain, excluded, graph, c)
            checked += 1
            applies = confirmation is not None and confirmation.claim_key == KEY and supported >= 1
            if d.outcome == REJECT:
                self.assertGreaterEqual(supported, 1, label)
                self.assertTrue(graph == "mismatch" or (applies and confirmation.status == "confirmed" and verified == 0 and graph != "match"), label)
            if d.outcome == ACCEPT:
                self.assertTrue(supported == 0 and uncertain == 0 and graph != "mismatch" and verified >= 1, label)
                self.assertGreaterEqual(verified + (graph == "match"), policy.QUORUM, label)
            if verified == 0 and supported == 0:
                self.assertEqual(d.outcome, UNCERTAIN, label)
            if d.needs_confirmation:
                self.assertTrue(d.outcome == UNCERTAIN and supported >= 1 and verified == 0 and not applies and graph == "", label)
                self.assertFalse(d.needs_escalation, label)
            if d.needs_escalation:
                self.assertEqual(d.outcome, UNCERTAIN, label)
                self.assertLess(len(js), policy.MAX_STRUCTURED_CALLS, label)
            if d.outcome == UNCERTAIN:
                self.assertTrue(d.requires_review and d.failure_categories, label)
            final = decision.publish(js, confirmation, g)  # what may be exported: never pending, always passes the gate
            self.assertEqual((final.consistency_status, final.needs_confirmation), ("ok", False), label)
            self.assertTrue(final.explanation.startswith(final.outcome), label)
        self.assertEqual(checked, 4 * 4 * 3 * 3 * 3 * 6)


class GateTests(unittest.TestCase):
    def test_clean_decisions_pass_the_gate(self):
        for js, conf in ((VER(2), None), (REJ(1), CONF("confirmed")), (REJ(1), None), (VER(1) + REJ(1), None), ([J(UNCERTAIN)], None), (VER(1), None)):
            d = decision.publish(js, conf)
            self.assertEqual((d.consistency_status, decision.check_consistency(d)), ("ok", []))
            self.assertTrue(d.explanation.startswith(d.outcome))

    def test_explanations_are_written_from_the_decision(self):
        accept_ = decision.publish(VER(2))
        self.assertIn("not verified by an independent structural tool", accept_.explanation)
        confirmed = decision.publish(REJ(1), CONF("confirmed"))
        for part in ("REJECT (confirmed_reject)", "connectivity", '"x"', "ref obs", "ann obs", "Confirmed by a second Gemini check"):
            self.assertIn(part, confirmed.explanation)
        refuted = decision.publish(REJ(1), CONF("refuted"))
        for part in ("UNCERTAIN (checks_disagree)", "did not confirm", "ref obs", "Review required."):
            self.assertIn(part, refuted.explanation)
        contested = decision.publish(VER(1) + REJ(1))
        self.assertIn("connectivity", contested.explanation)
        self.assertIn("1 call(s) judged the drawings equivalent", contested.explanation)

    def test_a_reject_that_never_got_its_confirming_call_is_exported_as_not_run(self):
        d = decision.publish(REJ(1), None, final=True)
        self.assertEqual((d.outcome, d.rule, d.requires_review, d.needs_confirmation), (UNCERTAIN, "confirmation_not_run", True, False))
        self.assertIn("confirming call was not run", d.explanation)
        self.assertTrue(decision.decide(REJ(1), None).needs_confirmation)  # while the run is open it is still pending

    def test_contradictions_are_caught_and_never_exported_as_clean(self):
        def confirmed_reject():
            return decision.decide(REJ(1), CONF("confirmed"))

        builders = {
            "accept_with_evidence": lambda: _tamper(decision.decide(VER(2)), evidence=[dict(SUPPORTED, votes=1)]),
            "reject_unconfirmed": lambda: _tamper(confirmed_reject(), evidence_status="asserted"),
            "reject_flagged_for_review": lambda: _tamper(confirmed_reject(), requires_review=True),
            "reject_with_verified_accept": lambda: _tamper(confirmed_reject(), verified_accepts=1),
            "reject_without_confirmation_record": lambda: _tamper(confirmed_reject(), confirmation=None),
            "reject_without_complete_evidence": lambda: (lambda d: _tamper(d, evidence=[dict(d.evidence[0], location="")]))(confirmed_reject()),
            "uncertain_not_flagged": lambda: _tamper(decision.decide(VER(1)), requires_review=False),
            "uncertain_without_cause": lambda: _tamper(decision.decide(VER(1)), failure_categories=[]),
            "vote_count_mismatch": lambda: (lambda d: _tamper(d, votes={**d.votes, "ACCEPT": 5}))(decision.decide(VER(2))),
            "unconfirmed_pending_at_export": lambda: decision.decide(REJ(1)),  # a reject still waiting for its call must not reach an export
        }
        for code, build in builders.items():
            done = decision.finalize(build())
            self.assertEqual((done.outcome, done.rule, done.consistency_status, done.requires_review), (UNCERTAIN, "consistency_conflict", "conflict", True), code)
            self.assertIn(code, done.consistency, code)
            self.assertIn(code, done.failure_categories, code)
            self.assertTrue(done.explanation.startswith("UNCERTAIN (consistency_conflict)") and code in done.explanation, code)

    def test_an_explanation_that_disagrees_with_the_outcome_is_caught(self):
        d = decision.decide(VER(2))
        d.explanation = "REJECT (confirmed_reject): something"
        self.assertTrue({"explanation_outcome_mismatch", "explanation_contradicts_outcome"} <= set(decision.check_consistency(d)))
        d = decision.decide(REJ(1), CONF("confirmed"))
        d.explanation = "REJECT (confirmed_reject): a difference was reported."  # names neither category nor location
        self.assertIn("explanation_missing_evidence", decision.check_consistency(d))


def _tamper(d, **fields):
    for k, v in fields.items():
        setattr(d, k, v)
    return d


class RecordedCallsTests(unittest.TestCase):
    """Behaviour of the rules on real recorded policy-prompt calls. No chemical truth is asserted."""

    @staticmethod
    def judgements(key):
        return [decision.judge_structured(rec(c["raw_text"])) for c in CALLS[key]]

    def test_no_recorded_task_publishes_a_reject_without_a_confirming_call(self):
        for key in CALLS:
            d = decision.publish(self.judgements(key), None)
            self.assertNotEqual(d.outcome, REJECT, key)
            self.assertEqual(d.consistency_status, "ok", key)

    def test_states_follow_the_votes(self):
        for key in CALLS:
            js = self.judgements(key)
            verified = sum(1 for j in js if j.vote == ACCEPT and j.verified)
            supported = sum(1 for j in js if j.vote == REJECT)
            d = decision.publish(js, None)
            if supported and verified:
                self.assertEqual(d.rule, "contested_reject", key)
            elif supported:
                self.assertEqual(d.rule, "confirmation_not_run", key)
            elif verified >= policy.QUORUM:
                self.assertEqual(d.rule, "verified_equivalence", key)

    def test_the_cases_under_review_behave_as_the_rules_say(self):
        expected = {"google30/molecule_316": "contested_reject", "google30/molecule_475": "verified_equivalence",
                    "google30/molecule_144": "confirmation_not_run", "google30/molecule_79": "confirmation_not_run",
                    "google30/molecule_172": "contested_reject", "google30/molecule_249": "contested_reject",
                    "google30/molecule_2": "confirmation_not_run", "google30/molecule_191": "confirmation_not_run",
                    "google30/molecule_48": "confirmation_not_run", "google10/molecule_470": "contested_reject",
                    "google10/molecule_118": "confirmation_not_run"}
        for key, rule in expected.items():
            self.assertEqual(decision.publish(self.judgements(key), None).rule, rule, key)
        for key in ("google30/molecule_144", "google30/molecule_79", "google30/molecule_2"):
            d = decision.decide(self.judgements(key))
            self.assertEqual((d.rule, d.needs_confirmation), ("reject_pending_confirmation", True), key)

    def test_a_confirming_call_decides_a_pending_reject(self):
        for key in ("google30/molecule_144", "google30/molecule_79"):
            js = self.judgements(key)
            claim = decision.decide(js).claim
            ck = decision.claim_key(claim["category"], claim["location"])
            confirmed = decision.publish(js, Confirmation("confirmed", ck, {"reference": "obs a", "annotated": "obs b"}))
            self.assertEqual((confirmed.outcome, confirmed.evidence_status, confirmed.consistency_status), (REJECT, "confirmed", "ok"), key)
            self.assertIn(claim["category"], confirmed.explanation)
            refuted = decision.publish(js, Confirmation("refuted", ck, {"reference": "obs a", "annotated": "obs a"}))
            self.assertEqual((refuted.outcome, refuted.rule), (UNCERTAIN, "checks_disagree"), key)

    def test_recorded_accepts_are_unchanged(self):
        for key in ("google30/molecule_484", "google30/molecule_475", "google10/molecule_25", "google10/molecule_132",
                    "google10/molecule_353", "google10/molecule_213"):
            d = decision.publish(self.judgements(key), None)
            self.assertEqual((d.outcome, d.rule, d.evidence), (ACCEPT, "verified_equivalence", []), key)


class GraphCheckTests(unittest.TestCase):
    def test_missing_smiles_is_not_provided(self):
        self.assertEqual(graphcheck.compare_smiles("", "CCO"), graphcheck.NOT_PROVIDED)
        self.assertEqual(graphcheck.compare_smiles("CCO", None), graphcheck.NOT_PROVIDED)

    @unittest.skipIf(graphcheck.rdkit_available(), "RDKit is installed")
    def test_without_rdkit_the_result_is_unavailable(self):
        self.assertEqual(graphcheck.compare_smiles("CCO", "OCC").status, "unavailable")

    @unittest.skipUnless(graphcheck.rdkit_available(), "RDKit is not installed")
    def test_rdkit_comparison(self):
        self.assertEqual(graphcheck.compare_smiles("CC(=O)Oc1ccccc1C(=O)O", "OC(=O)c1ccccc1OC(C)=O").status, "match")
        stereo = graphcheck.compare_smiles("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O")
        self.assertEqual((stereo.status, stereo.categories), ("mismatch", ("stereochemistry",)))
        self.assertEqual(graphcheck.compare_smiles("CCO", "CCN").status, "mismatch")
        self.assertEqual(graphcheck.compare_smiles("CCO", "not a smiles").status, "unparseable")

    def test_load_smiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.csv"
            path.write_text("IDs,Samples_SMILES,Annotated_SMILES\nmolecule_1,CCO,OCC\n", encoding="utf-8")
            self.assertEqual(graphcheck.load_smiles(str(path)), {"molecule_1": ("CCO", "OCC")})
            path.write_text("IDs,Samples\nmolecule_1,x\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                graphcheck.load_smiles(str(path))


if __name__ == "__main__":
    unittest.main()
