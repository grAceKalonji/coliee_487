"""Stage 1: turn raw COLIEE case files into clean, ready-to-search data.

Run:  python -m src.preprocess
Writes:
    data/processed/train_cases.json   {case_id: {"tokens": [...], "year": 2011, "contexts": [[...], ...]}}
    data/processed/test_cases.json    same shape
    data/processed/split.json         {"train": [query ids], "dev": [query ids]}
"""
import json
import random
import re
from pathlib import Path

# ---- shared paths + helpers (other stages import these) ----
DATA = Path("data")
PROCESSED = DATA / "processed"
RUNS = Path("runs")

TRAIN_CASES = DATA / "task1_train_files_2026" / "cases"
TRAIN_LABELS = DATA / "task1_train_files_2026" / "clean_task1_train_labels_2026.json"
TEST_CASES = DATA / "task1_test_files_2026"
TEST_QUERIES = DATA / "task1_test_no_labels_2026.json"
TEST_LABELS = DATA / "task1_test_labels_2026.json"

SUPPRESSED = "<FRAGMENT_SUPPRESSED>"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


# ---- stage 1 functions (TODO) ----

def read_case(path):
    """Return the raw text of one case file."""
    # TODO (Jack): open with encoding="utf-8", errors="ignore"
    raise NotImplementedError


def tokenize(text):
    """Lowercase, drop punctuation and stopwords, return a list of words.

    Keep SUPPRESSED out of the tokens (it's a marker, not a word).
    """
    # TODO (Jack)
    raise NotImplementedError


def extract_year(text):
    """Best guess at the year the case was decided, or None.

    Idea: take the most common 4-digit year (1900-2026) in the first ~2,000 characters.
    Used later to drop candidates decided after the query.
    """
    # TODO (Jack)
    raise NotImplementedError


def extract_citation_contexts(text, window=50):
    """Return one token list per <FRAGMENT_SUPPRESSED>: `window` words before + after it.

    These short snippets are the text the judge wrote right next to the citation,
    so they say what the cited case is about.
    """
    # TODO (Grace)
    raise NotImplementedError


def make_dev_split(query_ids, dev_fraction=0.2, seed=2026):
    """Shuffle the train query ids and hold out `dev_fraction` of them for tuning."""
    ids = sorted(query_ids)
    random.Random(seed).shuffle(ids)
    n_dev = int(len(ids) * dev_fraction)
    return {"train": sorted(ids[n_dev:]), "dev": sorted(ids[:n_dev])}


def process_folder(folder):
    """Run read_case -> tokenize / extract_year / extract_citation_contexts on every .txt."""
    out = {}
    for path in sorted(Path(folder).glob("*.txt")):
        text = read_case(path)
        out[path.name] = {
            "tokens": tokenize(text),
            "year": extract_year(text),
            "contexts": extract_citation_contexts(text),
        }
    return out


def main():
    save_json(process_folder(TRAIN_CASES), PROCESSED / "train_cases.json")
    save_json(process_folder(TEST_CASES), PROCESSED / "test_cases.json")
    save_json(make_dev_split(load_json(TRAIN_LABELS).keys()), PROCESSED / "split.json")
    print("done: data/processed/")


if __name__ == "__main__":
    main()
