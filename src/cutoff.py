"""Stage 5: decide how many candidates to return per query, then write the submission.

Returning too many kills precision, too few kills recall. Micro F1 needs a balance.
Run:  python -m src.cutoff
Writes:
    runs/pred_{dev,test}.json        {query_id: [candidate_id, ...]}   (same shape as labels)
    runs/submission_test.txt         COLIEE format: "query_id candidate_id run_name" per line
"""
from src.preprocess import RUNS, load_json, save_json

RUN_NAME = "GKJB"
TOP_N = 5          # simple rule: keep the top N        (tune on dev)
RATIO = 0.8        # dynamic rule: keep scores >= RATIO * best score (tune on dev)


def cut_top_n(run, n=TOP_N):
    return {q: [c for c, _ in ranked[:n]] for q, ranked in run.items()}


def cut_dynamic(run, ratio=RATIO, max_n=10):
    """Keep every candidate within `ratio` of the top score (at least 1, at most max_n)."""
    # TODO (Jack)
    raise NotImplementedError


def tune(dev_run, dev_labels):
    """Try a grid of N / ratio values on dev, return the best setting by micro F1."""
    # TODO (Jack): use src.evaluate.micro_f1
    raise NotImplementedError


def write_submission(pred, path, run_name=RUN_NAME):
    """One line per (query, candidate) in COLIEE's text format."""
    # TODO (Grace): strip ".txt" if the official format wants bare ids - check the task page
    raise NotImplementedError


def main():
    for split in ("dev", "test"):
        save_json(cut_top_n(load_json(RUNS / f"rerank_{split}.json")), RUNS / f"pred_{split}.json")
    write_submission(load_json(RUNS / "pred_test.json"), RUNS / "submission_test.txt")
    print("done: runs/pred_*.json, runs/submission_test.txt")


if __name__ == "__main__":
    main()
