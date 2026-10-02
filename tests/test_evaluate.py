import unittest

from src.evaluate import as_ids, micro_f1
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

    def test_as_ids_ranked_run(self):
        self.assertEqual(as_ids([["a", 3.0], ["b", 2.0], ["c", 1.0]], top=2), ["a", "b"])


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
