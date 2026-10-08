import tempfile
import unittest
from pathlib import Path

from src.preprocess import load_json, save_json
from src.validate_data import prepare_split, validate_training_data


class TestDataValidation(unittest.TestCase):
    def setUp(self):
        self.labels = {"q1": ["a"], "q2": ["b"]}
        self.cases = {"q1", "q2", "a", "b"}
        self.split = {"train": ["q1"], "dev": ["q2"]}

    def test_valid_data_keeps_full_candidate_pool(self):
        report = validate_training_data(self.labels, self.cases, self.split)
        self.assertEqual(report["case_count"], 4)
        self.assertEqual(report["relevant_pairs"], 2)

    def test_missing_query_or_candidate_is_rejected(self):
        for missing in ("q1", "a"):
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "Missing"):
                validate_training_data(self.labels, self.cases - {missing}, self.split)

    def test_invalid_splits_are_rejected(self):
        for split in (
            {"train": ["q1"], "dev": ["q1", "q2"]},
            {"train": ["q1", "q1"], "dev": ["q2"]},
            {"train": ["q1"], "dev": ["unknown"]},
            {"train": ["q1", "q2"], "dev": []},
        ):
            with self.subTest(split=split), self.assertRaises(ValueError):
                validate_training_data(self.labels, self.cases, split)

    def test_split_is_repeatable_and_conflicting_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cases = root / "cases"
            cases.mkdir()
            labels = {f"q{i}.txt": ["candidate.txt"] for i in range(10)}
            for name in [*labels, "candidate.txt"]:
                (cases / name).touch()
            label_path, split_path, audit_path = (root / n for n in
                                                   ("labels.json", "split.json", "audit.json"))
            save_json(labels, label_path)
            first = prepare_split(cases, label_path, split_path, audit_path)
            self.assertEqual(first["dev_queries"], 2)
            self.assertEqual(first, prepare_split(cases, label_path, split_path, audit_path))
            self.assertEqual(load_json(audit_path), first)
            save_json({"train": [], "dev": []}, split_path)
            with self.assertRaisesRegex(ValueError, "Existing split differs"):
                prepare_split(cases, label_path, split_path, audit_path)
            self.assertEqual(load_json(split_path), {"train": [], "dev": []})


if __name__ == "__main__":
    unittest.main()
