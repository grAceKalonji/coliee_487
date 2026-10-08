"""Validate Task 1 training data and save the dev split without preprocessing.

Run: python -m src.validate_data
Only training labels and case filenames are read; test labels are not used.
"""
from pathlib import Path

from src.preprocess import (
    PROCESSED, RUNS, TRAIN_CASES, TRAIN_LABELS, load_json, make_dev_split, save_json,
)

SEED = 2026
DEV_FRACTION = 0.2


def validate_training_data(labels, case_ids, split):
    """Reject missing files and invalid query splits before writing any output."""
    if not labels:
        raise ValueError("Training labels cannot be empty")
    if any(not isinstance(v, list) or any(not isinstance(c, str) for c in v)
           for v in labels.values()):
        raise ValueError("Task 1 labels must be lists of candidate filenames")
    train, dev = split["train"], split["dev"]
    if not train or not dev:
        raise ValueError("Train and dev query sets must both be nonempty")
    if len(train) != len(set(train)) or len(dev) != len(set(dev)):
        raise ValueError("Split contains duplicate query IDs")
    if set(train) & set(dev):
        raise ValueError("Train and dev queries overlap")
    if set(train) | set(dev) != set(labels):
        raise ValueError("Split must cover exactly the labeled training queries")
    relevant = {c for candidates in labels.values() for c in candidates}
    missing_queries = set(labels) - set(case_ids)
    missing_candidates = relevant - set(case_ids)
    if missing_queries or missing_candidates:
        raise ValueError(f"Missing query files: {sorted(missing_queries)}; "
                         f"missing relevant case files: {sorted(missing_candidates)}")
    return {
        "case_count": len(case_ids),
        "query_count": len(labels),
        "train_queries": len(train),
        "dev_queries": len(dev),
        "relevant_pairs": sum(len(set(v)) for v in labels.values()),
        "missing_query_files": 0,
        "missing_candidate_files": 0,
    }


def prepare_split(cases_folder=TRAIN_CASES, labels_path=TRAIN_LABELS,
                  split_path=PROCESSED / "split.json",
                  audit_path=RUNS / "task1_data_audit.json"):
    labels = load_json(labels_path)
    case_ids = {p.name for p in Path(cases_folder).glob("*.txt")}
    split = make_dev_split(labels, dev_fraction=DEV_FRACTION, seed=SEED)
    report = validate_training_data(labels, case_ids, split)
    split_path = Path(split_path)
    if split_path.exists() and load_json(split_path) != split:
        raise ValueError("Existing split differs from seed-2026 split; leaving it unchanged")
    report.update(seed=SEED, dev_fraction=DEV_FRACTION,
                  labels_path=str(labels_path), cases_folder=str(cases_folder),
                  split_path=str(split_path))
    if not split_path.exists():
        save_json(split, split_path)
    save_json(report, audit_path)
    return report


def main():
    report = prepare_split()
    print(f"cases={report['case_count']} queries={report['query_count']} "
          f"train={report['train_queries']} dev={report['dev_queries']} "
          f"relevant_pairs={report['relevant_pairs']}")
    print("Saved data/processed/split.json and runs/task1_data_audit.json")


if __name__ == "__main__":
    main()
