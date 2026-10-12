import json
import tempfile
import unittest
from pathlib import Path

from chemverify import gold, perception


def transcription(**over):
    base = {
        "drawing_style": "skeletal", "atom_labels": ["N", "OH", "Cl"], "abbreviations": [], "r_labels": ["R1"],
        "ring_sizes": [6, 6], "aromatic_ring_depiction": "circle", "double_bonds": 2, "triple_bonds": 0,
        "wedge_bonds": 1, "hashed_bonds": 0, "stereo_labels": [], "charges": [], "ink_colour": "black",
        "legibility": "clear", "substituent_description": "Cl at position 4.", "overlaps_or_ambiguities": [], "notes": "",
    }
    base.update(over)
    return base


class PerceptionTests(unittest.TestCase):
    def test_normalise_label(self):
        self.assertEqual(perception.normalise_label(" CH₃ "), "CH3")
        self.assertEqual(perception.normalise_label("2−"), "2-")
        self.assertNotEqual(perception.normalise_label("Ph"), perception.normalise_label("PH"))

    def test_multiset_f1(self):
        self.assertAlmostEqual(perception.multiset_f1(["a", "a", "b"], ["a", "b", "c"]), 2 / 3)
        self.assertEqual(perception.multiset_f1([], []), 1.0)
        self.assertEqual(perception.multiset_f1(["a"], []), 0.0)
        self.assertEqual(perception.multiset_f1(["a", "a"], ["a"]), 2 * (0.5 * 1.0) / 1.5)

    def test_validate(self):
        self.assertEqual(perception.validate(transcription()), [])
        bad = transcription(ink_colour="purple", double_bonds=-1, ring_sizes=["six"])
        del bad["legibility"]
        problems = " ".join(perception.validate(bad))
        for fragment in ("ink_colour", "double_bonds", "ring_sizes", "legibility"):
            self.assertIn(fragment, problems)
        self.assertEqual(perception.validate([1]), ["not a JSON object"])
        self.assertTrue(perception.validate(transcription(double_bonds=True)))

    def test_perfect_and_degraded_scores(self):
        gold_t = transcription()
        perfect = perception.score(transcription(), gold_t)
        self.assertEqual((perfect["overall"], perfect["critical"], perfect["problems"]), (1.0, 1.0, []))
        worse = perception.score(transcription(abbreviations=["PH"], atom_labels=["N", "Cl"], wedge_bonds=3), gold_t)
        self.assertLess(worse["critical"], 1.0)
        self.assertEqual(worse["fields"]["wedge_bonds"], 0.0)
        self.assertEqual(worse["within_one"]["wedge_bonds"], 0.0)
        near = perception.score(transcription(double_bonds=3), gold_t)
        self.assertEqual((near["fields"]["double_bonds"], near["within_one"]["double_bonds"]), (0.0, 1.0))

    def test_unparseable_prediction_scores_zero_on_populated_fields(self):
        s = perception.score(None, transcription())
        self.assertTrue(s["problems"])
        self.assertEqual(s["fields"]["atom_labels"], 0.0)
        self.assertEqual(s["fields"]["ring_sizes"], 0.0)

    def test_summarise_by_role(self):
        gold_t = transcription()
        scored = []
        for key, role, pred in (("m1/ref", "ref", transcription()), ("m1/wrong", "wrong", transcription(atom_labels=[]))):
            s = perception.score(pred, gold_t)
            s.update(image_key=key, role=role, molecule="m1", run_idx=0)
            scored.append(s)
        summary = perception.summarise(scored, n_boot=50)
        self.assertEqual(summary["ref"]["critical"]["mean"], 1.0)
        self.assertLess(summary["wrong"]["critical"]["mean"], 1.0)
        self.assertEqual(summary["all"]["images"], 2)
        text = perception.render(summary, {"name": "x", "prompt_name": "p", "prompt_sha": "s", "settings": {"model": "m"}})
        self.assertIn("critical", text)

    def test_schema_lists_every_field_once(self):
        schema = perception.json_schema()
        self.assertEqual(len(schema["properties"]), len(set(schema["properties"])))
        self.assertNotIn("notes", schema["required"])
        self.assertEqual(set(transcription()) - set(schema["properties"]), set())


def molecule_obj(mid="m1", errors=None, needs_review=False):
    default = [{"category": "missing_atom_or_group", "description": "methyl missing", "location": "C3", "severity": "high"}]
    return {
        "molecule": mid, "transcriptions": {r: transcription() for r in ("ref", "wrong", "correct")},
        "errors_in_wrong": default if errors is None else errors, "correct_version_issues": [],
        "primary_error_category": "missing_atom_or_group", "confidence": "high", "needs_user_review": needs_review,
        "review_notes": "check stereo" if needs_review else "",
    }


def verification_obj(mid="m1", verdicts=("confirmed",), missed=(), disputes=()):
    return {
        "molecule": mid, "status": "confirmed",  # the agent's own word; merge recomputes it
        "claim_checks": [{"claim": f"claim {i}", "verdict": v, "evidence": "seen"} for i, v in enumerate(verdicts)],
        "missed_differences": list(missed), "transcription_disputes": list(disputes), "notes": "",
    }


MISSED = {"category": "bond_order", "description": "double bond drawn single", "location": "ring", "severity": "medium"}


class GoldTests(unittest.TestCase):
    def test_validate_molecule(self):
        self.assertEqual(gold.validate_molecule(molecule_obj(), "m1"), [])
        self.assertTrue(gold.validate_molecule(molecule_obj("other"), "m1"))
        bad = molecule_obj()
        bad["errors_in_wrong"][0]["category"] = "made_up"
        bad["confidence"] = "certain"
        problems = " ".join(gold.validate_molecule(bad, "m1"))
        self.assertIn("category", problems)
        self.assertIn("confidence", problems)

    def test_export_csv_has_a_row_for_every_molecule(self):
        import csv

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "by_molecule").mkdir()
            (root / "verification").mkdir()
            good = molecule_obj("m1", errors=[
                {"category": "substituent_position", "description": "Cl meta, not para", "severity": "medium"},
                {"category": "abbreviation_changed", "description": "OMe written OCH3", "severity": "high"}])
            good["primary_error_category"] = "substituent_position"
            (root / "by_molecule" / "m1.json").write_text(json.dumps(good))
            (root / "by_molecule" / "m2.json").write_text(json.dumps(molecule_obj("m2", needs_review=True)))
            (root / "by_molecule" / "m3.json").write_text(json.dumps(molecule_obj("other")))  # wrong molecule id
            (root / "verification" / "m1.json").write_text(json.dumps(verification_obj("m1")))
            out = root / "out.csv"
            counts = gold.export_csv(root / "by_molecule", out, ["m1", "m2", "m3", "m4"])
            self.assertEqual(counts, {"adjudicated": 2, "pending": 1, "invalid": 1})
            with open(out, encoding="utf-8-sig", newline="") as fh:
                rows = {r["molecule"]: r for r in csv.DictReader(fh)}
            self.assertEqual(list(rows), ["m1", "m2", "m3", "m4"])
            m1 = rows["m1"]
            self.assertEqual((m1["status"], m1["verification"], m1["needs_human_review"], m1["worst_severity"]), ("adjudicated", "confirmed", "no", "high"))
            self.assertEqual(m1["all_error_categories"], "abbreviation_changed; substituent_position")
            self.assertIn("[substituent_position, medium] Cl meta, not para | [abbreviation_changed, high]", m1["errors_in_wrong_drawing"])
            self.assertEqual(m1["issues_in_corrected_drawing"], "none")
            self.assertEqual((rows["m2"]["needs_human_review"], rows["m2"]["verification"]), ("yes", "unverified"))
            self.assertEqual(rows["m3"]["status"], "invalid")
            self.assertEqual(rows["m4"]["status"], "pending")

    def test_validate_transcriptions(self):
        good = {"aaa": transcription(), "bbb": transcription(ink_colour="blue")}
        self.assertEqual(gold.validate_transcriptions(good), [])
        self.assertEqual(gold.validate_transcriptions(good, ["aaa", "bbb"]), [])
        self.assertIn("ccc: no transcription", gold.validate_transcriptions(good, ["aaa", "bbb", "ccc"]))
        self.assertIn("bbb: not one of the images you were given", gold.validate_transcriptions(good, ["aaa"]))
        bad = {"aaa": transcription(ink_colour="purple")}
        self.assertIn("aaa: ink_colour", " ".join(gold.validate_transcriptions(bad)))
        self.assertTrue(gold.validate_transcriptions([1]))

    def test_no_errors_requires_a_review_flag(self):
        self.assertTrue(gold.validate_molecule(molecule_obj(errors=[]), "m1"))
        self.assertEqual(gold.validate_molecule(molecule_obj(errors=[], needs_review=True), "m1"), [])

    def test_validate_verification(self):
        self.assertEqual(gold.validate_verification(verification_obj(), "m1"), [])
        self.assertTrue(gold.validate_verification(verification_obj("zzz"), "m1"))
        self.assertTrue(gold.validate_verification(verification_obj(verdicts=("maybe",)), "m1"))
        bad = verification_obj()
        bad["transcription_disputes"] = "none"
        self.assertTrue(gold.validate_verification(bad, "m1"))

    def test_verification_status_is_recomputed_from_evidence(self):
        self.assertEqual(gold.verification_status(verification_obj()), "confirmed")
        self.assertEqual(gold.verification_status(verification_obj(verdicts=("confirmed", "uncertain"))), "disputed")
        self.assertEqual(gold.verification_status(verification_obj(verdicts=("refuted",))), "disputed")
        self.assertEqual(gold.verification_status(verification_obj(missed=[MISSED])), "disputed")
        self.assertEqual(gold.verification_status(verification_obj(disputes=[{"role": "ref"}])), "disputed")

    def test_merge_combines_both_files_and_reports_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            by, ver = root / "by_molecule", root / "verification"
            by.mkdir()
            ver.mkdir()

            def put(folder, name, obj):
                (folder / f"{name}.json").write_text(json.dumps(obj) if not isinstance(obj, str) else obj, encoding="utf-8")

            put(by, "m1", molecule_obj("m1"))
            put(ver, "m1", verification_obj("m1"))
            put(by, "m2", molecule_obj("m2"))
            put(ver, "m2", verification_obj("m2", verdicts=("refuted",), missed=[MISSED]))  # agent says 'confirmed': overridden
            put(by, "m3", "{not json")
            put(by, "m4", {"molecule": "m4"})
            put(by, "m6", molecule_obj("m6"))  # no verification file at all
            summary = gold.merge(by, root, ["m1", "m2", "m3", "m4", "m5", "m6"])
            self.assertEqual(summary["ok"], ["m1", "m2", "m6"])
            self.assertEqual(summary["missing"], ["m5"])
            self.assertEqual(sorted(summary["invalid"]), ["m3", "m4"])
            self.assertEqual(summary["unverified"], ["m6"])
            self.assertEqual(summary["needs_review"], ["m2", "m6"])
            trans = json.loads((root / "transcriptions.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(trans), [f"m{i}/{r}" for i in (1, 2, 6) for r in ("correct", "ref", "wrong")])
            errors = json.loads((root / "errors.json").read_text(encoding="utf-8"))
            self.assertEqual(errors["m1"]["primary"], "missing_atom_or_group")
            self.assertEqual(errors["m1"]["verification"]["status"], "confirmed")
            self.assertEqual(errors["m2"]["verification"]["status"], "disputed")
            self.assertTrue(errors["m2"]["needs_user_review"])
            self.assertEqual(errors["m6"]["verification"]["status"], "unverified")
            table = (root / "review_table.md").read_text(encoding="utf-8")
            self.assertIn("methyl missing", table)
            self.assertIn("disputed (1 claim(s) not confirmed, 1 missed", table)
            self.assertIn("## Verifier findings", table)
            self.assertIn("double bond drawn single", table)

    def test_merge_reports_invalid_verification_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "by_molecule").mkdir()
            (root / "verification").mkdir()
            (root / "by_molecule" / "m1.json").write_text(json.dumps(molecule_obj("m1")), encoding="utf-8")
            (root / "verification" / "m1.json").write_text(json.dumps({"molecule": "m1"}), encoding="utf-8")
            summary = gold.merge(root / "by_molecule", root, ["m1"])
            self.assertEqual(summary["ok"], [])
            self.assertIn("verification", " ".join(summary["invalid"]["m1"]))

    def test_prepare_blind_hides_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = {}
            for m in ("m1", "m2"):
                images[m] = {}
                for role in ("ref", "wrong", "correct"):
                    path = root / "images" / m / f"{role}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"{m}-{role}".encode())
                    images[m][role] = {"file": f"images/{m}/{role}.png"}
            tasks = gold.prepare_blind(images, root, root / "gold")
            self.assertEqual(len(tasks), 6)
            self.assertEqual({(t["molecule"], t["role"]) for t in tasks}, {(m, r) for m in ("m1", "m2") for r in ("ref", "wrong", "correct")})
            for t in tasks:
                name = Path(t["path"]).name
                for word in ("ref", "wrong", "correct", "m1", "m2"):
                    self.assertNotIn(word, name)
                self.assertTrue(Path(t["path"]).exists())
            mapping = json.loads((root / "gold" / "blind_map.json").read_text(encoding="utf-8"))
            self.assertEqual(len(mapping), 6)
            again = gold.prepare_blind(images, root, root / "gold")
            self.assertEqual([t["token"] for t in again], [t["token"] for t in tasks])


if __name__ == "__main__":
    unittest.main()
