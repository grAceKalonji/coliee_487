"""Score predictions with COLIEE's metric: micro precision, recall, F1.

Run:
    python -m src.evaluate runs/pred_dev.json            # dev queries from data/processed/split.json
    python -m src.evaluate runs/pred_test.json --test    # test (final numbers only!)

Accepts either a prediction file {q: [cand, ...]} or a ranked run {q: [[cand, score], ...]}
(for a ranked run, add --top N to cut it first).
Missing queries count as empty predictions; unexpected query IDs are rejected.
"""
import argparse
import sys

from src.preprocess import PROCESSED, TEST_LABELS, TRAIN_LABELS, load_json


def as_ids(items, top=None):
    """Turn [[cand, score], ...] or [cand, ...] into a list of candidate ids."""
    if top is not None and top <= 0:
        raise ValueError("top must be a positive integer")
    ids = [x[0] if isinstance(x, (list, tuple)) else x for x in items]
    return ids[:top] if top is not None else ids


def micro_f1(pred, labels):
    """Micro-averaged scores over the expected queries in `labels`.

    tp = predicted and correct, fp = predicted but wrong, fn = correct but missed.
    Totals are summed over all queries first, then P/R/F1 are computed once.
    """
    unexpected = set(pred) - set(labels)
    if unexpected:
        raise ValueError(f"Unexpected prediction query IDs: {sorted(unexpected)}")
    tp = fp = fn = 0
    for q, relevant in labels.items():
        gold = set(relevant)
        predicted = set(pred.get(q, []))
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main(argv):
    if not argv:
        print(__doc__)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions")
    parser.add_argument("--test", action="store_true", help="Score final test predictions")
    parser.add_argument("--top", type=int, help="Keep a positive number of candidates per query")
    parser.add_argument("--split", default=str(PROCESSED / "split.json"),
                        help="Saved train/dev split (dev evaluation only)")
    args = parser.parse_args(argv)
    if args.top is not None and args.top <= 0:
        parser.error("--top must be a positive integer")
    try:
        labels = load_json(TEST_LABELS if args.test else TRAIN_LABELS)
        if not args.test:
            dev_ids = load_json(args.split)["dev"]
            if not dev_ids or len(dev_ids) != len(set(dev_ids)):
                raise ValueError("Dev split must contain unique query IDs and cannot be empty")
            unknown = set(dev_ids) - set(labels)
            if unknown:
                raise ValueError(f"Dev query IDs missing from labels: {sorted(unknown)}")
            labels = {q: labels[q] for q in dev_ids}
        pred = {q: as_ids(items, args.top) for q, items in load_json(args.predictions).items()}
        s = micro_f1(pred, labels)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    missing = len(set(labels) - set(pred))
    print(f"expected={len(labels)} predicted={len(pred)} missing={missing}  "
          f"P={s['precision']:.4f}  R={s['recall']:.4f}  F1={s['f1']:.4f}  "
          f"(tp={s['tp']} fp={s['fp']} fn={s['fn']})")


if __name__ == "__main__":
    main(sys.argv[1:])
