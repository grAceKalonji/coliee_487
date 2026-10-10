import tempfile
import unittest
from pathlib import Path

from src.preprocess import extract_year, read_case, tokenize


class TestPreprocess(unittest.TestCase):
    def test_read_case_utf8_and_invalid_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "case.txt"
            path.write_bytes("Court’s decision\n".encode("utf-8") + b"\xffappeal")
            self.assertEqual(read_case(path), "Court’s decision\nappeal")

    def test_tokenize_removes_marker_punctuation_and_stopwords(self):
        text = "The COURT cited <FRAGMENT_SUPPRESSED>, and dismissed appeal 123."
        self.assertEqual(tokenize(text), ["court", "cited", "dismissed", "appeal", "123"])

    def test_tokenize_empty_text(self):
        self.assertEqual(tokenize(""), [])

    def test_tokenize_removes_reference_marker(self):
        self.assertEqual(tokenize("See REFERENCE_SUPPRESSED for costs."), ["see", "costs"])

    def test_tokenize_removes_all_marker_variants(self):
        text = "per FRAGMENT_SUPPRESSED, <CITATION_SUPPRESSED> on DATE_SUPPRESSED appeal"
        self.assertEqual(tokenize(text), ["per", "appeal"])

    def test_extract_year_uses_latest_year(self):
        self.assertEqual(extract_year("SC 2001, c 27. Decision dated March 16, 2012. Citing 1999."), 2012)

    def test_extract_year_ignores_out_of_range_and_partial_years(self):
        self.assertIsNone(extract_year("1899 2027 2030 12011 20110"))
        self.assertEqual(extract_year("1900 and 2026"), 2026)

    def test_extract_year_reads_whole_case(self):
        self.assertEqual(extract_year("2005 " + "x" * 5000 + " 2011"), 2011)

    def test_extract_year_without_dates(self):
        self.assertIsNone(extract_year("No date in this case."))


if __name__ == "__main__":
    unittest.main()
