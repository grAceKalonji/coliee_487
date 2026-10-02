"""Score predictions with COLIEE's metric: micro precision, recall, F1.

Run:
    python -m src.evaluate runs/pred_dev.json            # dev  (labels from train file)
    python -m src.evaluate runs/pred_test.json --test    # test (final numbers only!)

Accepts either a prediction file {q: [cand, ...]} or a ranked run {q: [[cand, score], ...]}
(for a ranked run, add --top N to cut it first).
"""
import sys

from src.preprocess import TEST_LABELS, TRAIN_LABELS, load_json


def as_ids(items, top=None):
    """Turn [[cand, score], ...] or [cand, ...] into a list of candidate ids."""
    ids = [x[0] if isinstance(x, (list, tuple)) else x for x in items]
    return ids[:top] if top else ids


def micro_f1(pred, labels):
    """Micro-averaged scores over the queries that appear in `pred`.

    tp = predicted and correct, fp = predicted but wrong, fn = correct but missed.
    Totals are summed over all queries first, then P/R/F1 are computed once.
    """
    tp = fp = fn = 0
    for q, predicted in pred.items():
        gold = set(labels.get(q, []))
        predicted = set(predicted)
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
    top = int(argv[argv.index("--top") + 1]) if "--top" in argv else None
    labels = load_json(TEST_LABELS if "--test" in argv else TRAIN_LABELS)
    pred = {q: as_ids(items, top) for q, items in load_json(argv[0]).items()}
    s = micro_f1(pred, labels)
    print(f"queries={len(pred)}  P={s['precision']:.4f}  R={s['recall']:.4f}  F1={s['f1']:.4f}  "
          f"(tp={s['tp']} fp={s['fp']} fn={s['fn']})")


if __name__ == "__main__":
    main(sys.argv[1:])
