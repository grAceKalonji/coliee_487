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
from collections import Counter
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
# cases also write citations as FRAGMENT_SUPPRESSED, REFERENCE_SUPPRESSED or CITATION_SUPPRESSED,
# sometimes without the brackets or glued to the next word
CITATION_MARK = re.compile(r"<?[A-Z]*(?:FRAGMENT|REFERENCE|CITATION)_SUPPRESSED>?")
# DATE_SUPPRESSED is a blanked-out date, not a citation: dropped from tokens, no context
DATE_MARK = re.compile(r"<?DATE_SUPPRESSED>?")

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for
from further had has have having he her here hers herself him himself his how i if in
into is it its itself just me more most my myself no nor not now of off on once only or
other our ours ourselves out over own same she should so some such than that the their
theirs them themselves then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your
yours yourself yourselves
""".split())

WORD = re.compile(r"[a-z0-9]+")
YEAR = re.compile(r"\b(19\d{2}|20[0-2]\d)\b")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


# ---- stage 1 functions ----

def read_case(path):
    """Return the raw text of one case file."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def tokenize(text):
    """Lowercase, drop punctuation and stopwords, return a list of words.

    Keep SUPPRESSED out of the tokens (it's a marker, not a word).
    """
    text = DATE_MARK.sub(" ", CITATION_MARK.sub(" ", text)).lower()
    return [w for w in WORD.findall(text) if w not in STOPWORDS]


def extract_year(text):
    """Best guess at the year the case was decided, or None.

    Idea: take the most common 4-digit year (1900-2026) in the first ~2,000 characters.
    Used later to drop candidates decided after the query.
    """
    years = [int(y) for y in YEAR.findall(text[:2000]) if int(y) <= 2026]
    if not years:
        return None
    return Counter(years).most_common(1)[0][0]


def extract_citation_contexts(text, window=50):
    """Return one token list per citation marker: `window` words before + after it.

    These short snippets are the text the judge wrote right next to the citation,
    so they say what the cited case is about.
    """
    # tokenize the pieces between markers, remembering where each marker fell
    words, marks = [], []
    for i, piece in enumerate(CITATION_MARK.split(text)):
        if i > 0:
            marks.append(len(words))
        words.extend(tokenize(piece))
    return [words[max(0, m - window):m + window] for m in marks]


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
