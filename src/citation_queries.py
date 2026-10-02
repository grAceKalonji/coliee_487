"""Stage 3: search with the text around each redacted citation instead of the whole case.

Each <FRAGMENT_SUPPRESSED> context becomes its own short query. A candidate's final
score combines its scores across all of the query case's contexts.
Run:  python -m src.citation_queries
Writes:  runs/ctx_{dev,test}.json   {query_id: [[candidate_id, score], ...]}
"""
from src.bm25 import TOP_K, apply_filters, build_index, score
from src.preprocess import PROCESSED, RUNS, TEST_QUERIES, load_json, save_json
from collections import Counter


def combine(per_context_scores, how="max"):
    """Merge a list of {case_id: score} dicts into one.

    how="max": a candidate's score = its best match to any single context.
                (one strong match is enough - a judge cites a case for one point)
    how="sum": add scores up (rewards cases that match many contexts).
    """
    # TODO (Grace)
    raise NotImplementedError


def rank(query_ids, cases, index, how="max"):
    run = {}
    for q in query_ids:
        contexts = cases[q]["contexts"] or [cases[q]["tokens"]]   # fall back to full text
        per_ctx = [score(Counter(c), index) for c in contexts]
        scores = apply_filters(q, combine(per_ctx, how), cases)
        run[q] = sorted(scores.items(), key=lambda x: -x[1])[:TOP_K]
    return run


def main():
    split = load_json(PROCESSED / "split.json")
    train = load_json(PROCESSED / "train_cases.json")
    save_json(rank(split["dev"], train, build_index(train)), RUNS / "ctx_dev.json")

    test = load_json(PROCESSED / "test_cases.json")
    test_queries = load_json(TEST_QUERIES).keys()
    save_json(rank(test_queries, test, build_index(test)), RUNS / "ctx_test.json")
    print("done: runs/ctx_*.json")


if __name__ == "__main__":
    main()
