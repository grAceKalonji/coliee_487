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
QUERY_TERMS = 100  # whole-case queries: keep only the 100 most distinctive words (None = all)
YEAR_SLACK = 1  # allow candidates up to 1 year newer than the query


def build_index(cases):
    """Inverted index: {"terms": {word: {case_id: count}}, "lengths": {case_id: n}, "avg_len": x, "N": n}."""
    terms, lengths = {}, {}
    for case_id, case in cases.items():
        lengths[case_id] = len(case["tokens"])
        # count each word once per case, then file the count under that word
        for word, count in Counter(case["tokens"]).items():
            terms.setdefault(word, {})[case_id] = count
    N = len(cases)
    return {"terms": terms, "lengths": lengths, "N": N,
            "avg_len": sum(lengths.values()) / N if N else 0.0}


def idf(N, df):
    """Rare words get big weights, common words small ones (never negative)."""
    return math.log((N - df + 0.5) / (df + 0.5) + 1)


def top_terms(query_counts, index, n=QUERY_TERMS):
    """Keep the query's n most distinctive words, ranked by tf * idf.

    A whole case has ~800 different words; most are common and just slow us down.
    Words not in the index are dropped (they can't match anything).
    """
    N, terms = index["N"], index["terms"]
    weights = {w: tf * idf(N, len(terms[w])) for w, tf in query_counts.items() if w in terms}
    if n is None or len(weights) <= n:
        return Counter({w: query_counts[w] for w in weights})
    best = sorted(weights, key=lambda w: (-weights[w], w))[:n]
    return Counter({w: query_counts[w] for w in best})


def score(query_counts, index, k1=None, b=None):
    """Return {case_id: BM25 score} for one query (term-at-a-time accumulators).

    query_counts: {word: count} for the query. Each distinct word counts once.
    """
    k1 = K1 if k1 is None else k1
    b = B if b is None else b
    N, terms = index["N"], index["terms"]
    lengths, avg_len = index["lengths"], index["avg_len"]
    acc = {}
    for word in query_counts:
        postings = terms.get(word)
        if not postings:
            continue  # word appears in no case: adds nothing
        w = idf(N, len(postings))
        for case_id, tf in postings.items():
            norm = 1 - b + b * lengths[case_id] / avg_len   # >1 for long cases, <1 for short
            acc[case_id] = acc.get(case_id, 0.0) + w * tf * (k1 + 1) / (tf + k1 * norm)
    return acc


def apply_filters(query_id, scores, cases):
    """Remove candidates we know can't be right.

    - the query itself
    - cases with a year later than the query's year + YEAR_SLACK (a case can't cite the future)
      Tested on Train: slack 1 loses only 2% of true citations and removes ~30% of the pool.
    - keep candidates whose year is None (don't guess)
    """
    # TODO (Jack)
    raise NotImplementedError


def rank(query_ids, cases, index):
    run = {}
    for q in query_ids:
        query = top_terms(Counter(cases[q]["tokens"]), index)
        scores = apply_filters(q, score(query, index), cases)
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
