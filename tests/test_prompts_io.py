import json
import tempfile
import unittest
from pathlib import Path

from chemverify import gold, runner, splits
from chemverify.prompts import load_prompt

BOM = "﻿"


class BomTolerance(unittest.TestCase):
    """PowerShell 5.1 and Notepad write a UTF-8 byte-order mark; nothing may choke on it or leak it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_prompt_bom_is_stripped_and_hash_is_stable(self):
        plain = self.root / "plain.txt"
        with_bom = self.root / "bom.txt"
        plain.write_text("You are a checker.\n", encoding="utf-8")
        with_bom.write_text(BOM + "You are a checker.\n\n", encoding="utf-8")
        a, b = load_prompt(plain), load_prompt(with_bom)
        self.assertEqual(b.text, "You are a checker.")
        self.assertNotIn(BOM, b.text)
        self.assertEqual(a.sha, b.sha)

    def test_empty_prompt_is_rejected(self):
        empty = self.root / "empty.txt"
        empty.write_text(BOM + "  \n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_prompt(empty)

    def test_split_and_records_with_bom(self):
        path = self.root / "splits.json"
        path.write_text(BOM + json.dumps({"dev": ["a"], "heldout": ["b"]}), encoding="utf-8")
        self.assertEqual(splits.load_split(path)["heldout"], ["b"])
        out = self.root / "outputs.jsonl"
        out.write_text(BOM + json.dumps({"key": "k1", "status": "ok"}) + "\n", encoding="utf-8")
        self.assertIn("k1", runner.load_records(out))

    def test_gold_files_with_bom(self):
        from .test_perception_gold import molecule_obj, verification_obj

        by, ver = self.root / "by_molecule", self.root / "verification"
        by.mkdir()
        ver.mkdir()
        (by / "m1.json").write_text(BOM + json.dumps(molecule_obj("m1")), encoding="utf-8")
        (ver / "m1.json").write_text(BOM + json.dumps(verification_obj("m1")), encoding="utf-8")
        summary = gold.merge(by, self.root, ["m1"])
        self.assertEqual((summary["ok"], summary["invalid"]), (["m1"], {}))


if __name__ == "__main__":
    unittest.main()
