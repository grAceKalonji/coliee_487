"""Stage 4: combine the full-text ranking (stage 2) and the context ranking (stage 3).

Two options, try RRF first (no training needed):
  - RRF: score = sum over rankings of 1 / (60 + rank)
  - LightGBM: learn how to weigh features (bm25 score, ctx score, ranks, year gap, ...)
Run:  python -m src.rerank
Writes:  runs/rerank_{dev,test}.json
"""
from src.preprocess import RUNS, load_json, save_json

RRF_K = 60


def rrf(runs, k=RRF_K):
    """Reciprocal Rank Fusion over several runs {query: [[cand, score], ...]}."""
    fused = {}
    for run in runs:
        for q, ranked in run.items():
            bucket = fused.setdefault(q, {})
            for rank, (cand, _) in enumerate(ranked, start=1):
                bucket[cand] = bucket.get(cand, 0.0) + 1.0 / (k + rank)
    return {q: sorted(s.items(), key=lambda x: -x[1]) for q, s in fused.items()}


def make_features(query_id, candidate_id, runs, cases):
    """One row of numbers describing (query, candidate) for LightGBM."""
    # TODO (Grace): bm25 score/rank, ctx score/rank, year gap, length ratio
    raise NotImplementedError


def train_lightgbm(train_queries, labels, runs, cases):
    """Fit an LGBMRanker (lambdarank) on train queries. Returns the model."""
    # TODO (Jack). Train its candidate runs need to exist for train queries too.
    raise NotImplementedError


def main():
    for split in ("dev", "test"):
        runs = [load_json(RUNS / f"bm25_{split}.json"), load_json(RUNS / f"ctx_{split}.json")]
        save_json(rrf(runs), RUNS / f"rerank_{split}.json")
    print("done: runs/rerank_*.json (RRF)")


if __name__ == "__main__":
    main()
