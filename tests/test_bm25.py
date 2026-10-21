import unittest
from collections import Counter

from src.bm25 import build_index, idf, score, top_terms

# toy pool: "visa" is rare (1 case), "court" is everywhere (3 cases)
CASES = {
    "a.txt": {"tokens": ["visa", "officer", "court"]},
    "b.txt": {"tokens": ["refugee", "court"]},
    "c.txt": {"tokens": ["refugee", "court", "appeal", "appeal", "costs", "costs"]},
}


class TestIndex(unittest.TestCase):
    def setUp(self):
        self.index = build_index(CASES)

    def test_shape(self):
        self.assertEqual(self.index["N"], 3)
        self.assertEqual(self.index["terms"]["appeal"], {"c.txt": 2})
        self.assertEqual(self.index["lengths"]["b.txt"], 2)
        self.assertAlmostEqual(self.index["avg_len"], 11 / 3)

    def test_empty_pool(self):
        self.assertEqual(build_index({})["N"], 0)


class TestScore(unittest.TestCase):
    def setUp(self):
        self.index = build_index(CASES)

    def test_idf_rare_beats_common_and_is_positive(self):
        self.assertGreater(idf(3, 1), idf(3, 3))
        self.assertGreater(idf(3, 3), 0)

    def test_matches_hand_calculation(self):
        # "visa" in a.txt: idf(3,1) * 1*(1.5+1) / (1 + 1.5*(1 - 0.75 + 0.75*3/(11/3)))
        norm = 1 - 0.75 + 0.75 * 3 / (11 / 3)
        expected = idf(3, 1) * 2.5 / (1 + 1.5 * norm)
        self.assertAlmostEqual(score(Counter(["visa"]), self.index)["a.txt"], expected)

    def test_rare_word_outranks_common_word(self):
        s = score(Counter(["visa", "court"]), self.index)
        self.assertEqual(max(s, key=s.get), "a.txt")

    def test_unknown_word_adds_nothing(self):
        self.assertEqual(score(Counter(["zebra"]), self.index), {})
        self.assertEqual(score(Counter(["visa", "zebra"]), self.index),
                         score(Counter(["visa"]), self.index))

    def test_longer_case_scores_lower_for_same_tf(self):
        # "refugee" appears once in b (short) and once in c (long)
        s = score(Counter(["refugee"]), self.index)
        self.assertGreater(s["b.txt"], s["c.txt"])

    def test_b_zero_turns_off_length_penalty(self):
        s = score(Counter(["refugee"]), self.index, b=0)
        self.assertAlmostEqual(s["b.txt"], s["c.txt"])


class TestTopTerms(unittest.TestCase):
    def test_keeps_most_distinctive_and_drops_unknown(self):
        index = build_index(CASES)
        q = Counter({"court": 1, "visa": 1, "zebra": 5})
        self.assertEqual(top_terms(q, index, n=1), Counter({"visa": 1}))
        self.assertEqual(set(top_terms(q, index, n=None)), {"court", "visa"})


if __name__ == "__main__":
    unittest.main()
