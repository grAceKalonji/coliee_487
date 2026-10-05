import unittest

from src.preprocess import SUPPRESSED, extract_citation_contexts, tokenize


class TestTokenize(unittest.TestCase):
    def test_lowercase_and_punctuation(self):
        self.assertEqual(tokenize("Officer, OFFICER. officer!"), ["officer"] * 3)

    def test_drops_stopwords(self):
        self.assertEqual(tokenize("the duty of the officer"), ["duty", "officer"])

    def test_drops_marker(self):
        self.assertEqual(tokenize(f"visa {SUPPRESSED} officer"), ["visa", "officer"])

    def test_drops_other_markers(self):
        text = "visa FRAGMENT_SUPPRESSEDand REFERENCE_SUPPRESSED, DATE_SUPPRESSED officer"
        self.assertEqual(tokenize(text), ["visa", "officer"])

    def test_empty(self):
        self.assertEqual(tokenize(""), [])


class TestCitationContexts(unittest.TestCase):
    def test_one_per_marker(self):
        text = f"visa officer {SUPPRESSED} duty fairness {SUPPRESSED} dialogue"
        self.assertEqual(len(extract_citation_contexts(text)), 2)

    def test_other_citation_markers(self):
        text = "visa REFERENCE_SUPPRESSED officer CITATION_SUPPRESSED duty DATE_SUPPRESSED"
        self.assertEqual(len(extract_citation_contexts(text)), 2)

    def test_no_markers(self):
        self.assertEqual(extract_citation_contexts("no citations here"), [])

    def test_window(self):
        text = f"a1 b1 c1 d1 {SUPPRESSED} e1 f1 g1"
        self.assertEqual(extract_citation_contexts(text, window=2), [["c1", "d1", "e1", "f1"]])

    def test_window_at_edges(self):
        self.assertEqual(extract_citation_contexts(f"{SUPPRESSED} visa", window=5), [["visa"]])

    def test_window_crosses_other_markers(self):
        # the second marker is skipped, not counted as a word
        text = f"visa {SUPPRESSED} officer {SUPPRESSED} duty"
        self.assertEqual(extract_citation_contexts(text, window=2),
                         [["visa", "officer", "duty"], ["visa", "officer", "duty"]])


if __name__ == "__main__":
    unittest.main()
