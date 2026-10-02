"""Stage 2: baseline. Score every candidate against the whole query case with BM25.

This is our benchmark (R0), modelled on TQM's 2024 run: lexical scoring + filters.
Run:  python -m src.bm25
Writes:  runs/bm25_{dev,test}.json   {query_id: [[candidate_id, score], ...]}  (top 100)
"""
import math
from collections import Counter

from src.preprocess import PROCESSED, RUNS, TEST_QUERIES, load_json, save_json

K1 = 1.5      # how fast repeated words stop helping   (tune on dev)
B = 0.75      # how much long cases get penalized       (tune on dev)
TOP_K = 100


def build_index(cases):
    """Inverted index: {"terms": {word: {case_id: count}}, "lengths": {case_id: n}, "avg_len": x, "N": n}."""
    # TODO (Grace): same shape as our COS487 index
    raise NotImplementedError


def idf(N, df):
    return math.log((N - df + 0.5) / (df + 0.5) + 1)


def score(query_tokens, index):
    """Return {case_id: BM25 score} for one query (term-at-a-time accumulators)."""
    # TODO (Grace)
    raise NotImplementedError


def apply_filters(query_id, scores, cases):
    """Remove candidates we know can't be right.

    - the query itself
    - cases with a year later than the query's year (a case can't cite the future)
    """
    # TODO (Jack)
    raise NotImplementedError


def rank(query_ids, cases, index):
    run = {}
    for q in query_ids:
        q_tokens = cases[q]["tokens"]
        scores = apply_filters(q, score(Counter(q_tokens), index), cases)
        run[q] = sorted(scores.items(), key=lambda x: -x[1])[:TOP_K]
    return run


def main():
    split = load_json(PROCESSED / "split.json")
    train = load_json(PROCESSED / "train_cases.json")
    save_json(rank(split["dev"], train, build_index(train)), RUNS / "bm25_dev.json")

    test = load_json(PROCESSED / "test_cases.json")
    test_queries = load_json(TEST_QUERIES).keys()
    save_json(rank(test_queries, test, build_index(test)), RUNS / "bm25_test.json")
    print("done: runs/bm25_*.json")


if __name__ == "__main__":
    main()
