import unittest
import io
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

from src.evaluate import as_ids, main, micro_f1
from src.preprocess import make_dev_split
from src.rerank import rrf


class TestMicroF1(unittest.TestCase):
    def test_perfect(self):
        s = micro_f1({"q1": ["a", "b"]}, {"q1": ["a", "b"]})
        self.assertEqual(s["f1"], 1.0)

    def test_micro_not_macro(self):
        # q1: 1 tp, 0 fp, 0 fn   q2: 0 tp, 1 fp, 3 fn  -> P=1/2, R=1/4
        s = micro_f1({"q1": ["a"], "q2": ["x"]}, {"q1": ["a"], "q2": ["b", "c", "d"]})
        self.assertAlmostEqual(s["precision"], 0.5)
        self.assertAlmostEqual(s["recall"], 0.25)
        self.assertAlmostEqual(s["f1"], 2 * 0.5 * 0.25 / 0.75)

    def test_empty_prediction(self):
        self.assertEqual(micro_f1({"q1": []}, {"q1": ["a"]})["f1"], 0.0)

    def test_missing_query_counts_as_false_negative(self):
        s = micro_f1({"q1": ["a"]}, {"q1": ["a"], "q2": ["b"]})
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (1, 0, 1))
        self.assertEqual(s["recall"], 0.5)

    def test_no_predictions_counts_all_gold(self):
        s = micro_f1({}, {"q1": ["a"], "q2": ["b", "c"]})
        self.assertEqual(s["fn"], 3)
        self.assertEqual(s["f1"], 0.0)

    def test_duplicate_candidates_count_once(self):
        s = micro_f1({"q": ["a", "a", "x", "x"]}, {"q": ["a", "a"]})
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (1, 1, 0))

    def test_unexpected_query_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            micro_f1({"unknown": []}, {"q": ["a"]})

    def test_top_must_be_positive(self):
        for top in (0, -1):
            with self.subTest(top=top), self.assertRaises(ValueError):
                as_ids(["a"], top=top)

    def test_as_ids_ranked_run(self):
        self.assertEqual(as_ids([["a", 3.0], ["b", 2.0], ["c", 1.0]], top=2), ["a", "b"])


class TestEvaluationCLI(unittest.TestCase):
    def test_dev_uses_saved_split_and_reports_missing_queries(self):
        output = io.StringIO()
        with patch("src.evaluate.load_json", side_effect=[
            {"train": ["x"], "q1": ["a"], "q2": ["b"]},
            {"train": ["train"], "dev": ["q1", "q2"]},
            {"q1": [["a", 9], ["wrong", 1]]},
        ]), redirect_stdout(output):
            main(["pred.json", "--top", "1"])
        self.assertIn("expected=2 predicted=1 missing=1", output.getvalue())
        self.assertIn("R=0.5000", output.getvalue())
        self.assertIn("F1=0.6667", output.getvalue())

    def test_test_mode_does_not_load_dev_split(self):
        with patch("src.evaluate.load_json", side_effect=[
            {"q": ["a"]}, {"q": ["a"]},
        ]) as load, redirect_stdout(io.StringIO()):
            main(["pred.json", "--test"])
        self.assertEqual(load.call_count, 2)

    def test_bad_dev_splits_are_rejected(self):
        for dev in ([], ["q", "q"], ["unknown"]):
            with self.subTest(dev=dev), patch("src.evaluate.load_json", side_effect=[
                {"q": ["a"]}, {"dev": dev},
            ]), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                main(["pred.json"])
            self.assertEqual(exc.exception.code, 2)

    def test_invalid_top_is_rejected_before_loading_data(self):
        with patch("src.evaluate.load_json") as load, redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as exc:
            main(["pred.json", "--top", "0"])
        self.assertEqual(exc.exception.code, 2)
        load.assert_not_called()


class TestHelpers(unittest.TestCase):
    def test_dev_split_disjoint_and_stable(self):
        ids = [f"{i:06d}.txt" for i in range(100)]
        s1, s2 = make_dev_split(ids), make_dev_split(ids)
        self.assertEqual(s1, s2)
        self.assertEqual(len(s1["dev"]), 20)
        self.assertFalse(set(s1["dev"]) & set(s1["train"]))

    def test_rrf_rewards_agreement(self):
        a = {"q": [["x", 9], ["y", 8]]}
        b = {"q": [["x", 5], ["z", 4]]}
        self.assertEqual(rrf([a, b])["q"][0][0], "x")


if __name__ == "__main__":
    unittest.main()
