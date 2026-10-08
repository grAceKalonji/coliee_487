# COLIEE 2026 Task 1: Legal Case Retrieval

Given a query case (a Federal Court of Canada decision), find the earlier cases it cites ("noticed cases") from a pool of candidate cases. Everything runs in the terminal.

Team: Grace Kalonji, Jack Bergin

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put the COLIEE files in `data/` (they are not in git):

```
data/
├── task1_train_files_2026/
│   ├── cases/                          # 7,708 .txt cases
│   └── clean_task1_train_labels_2026.json
├── task1_test_files_2026/              # 1,848 .txt cases
├── task1_test_labels_2026.json         # for final scoring only
└── task1_test_no_labels_2026.json
```

## Pipeline

Run from the repo root, in order:

```bash
python -m src.preprocess          # 1. clean cases, pull years and citation contexts, make the dev split
python -m src.bm25                # 2. baseline: BM25 over full text + filters
python -m src.citation_queries    # 3. BM25 using text around each <FRAGMENT_SUPPRESSED>
python -m src.rerank              # 4. combine rankings (RRF / LightGBM)
python -m src.cutoff              # 5. choose how many cases to return, write submission
python -m src.evaluate runs/<file>.json   # micro precision / recall / F1
```

Each stage reads the previous stage's output from `data/processed/` or `runs/`.

## Layout

```
src/        one file per pipeline stage
tests/      unit tests: python -m unittest discover tests
data/       COLIEE data + processed files (ignored)
runs/       rankings and submissions (ignored)
```

## Evaluation

Prepare and validate the Task 1 dev split without running unfinished preprocessing:

```bash
python -m src.validate_data
```

This checks training query and relevant-case files, saves the seed-2026 split in
`data/processed/split.json`, and writes counts to `runs/task1_data_audit.json`.
All training cases remain in the retrieval pool. Dev evaluation uses the saved
dev query IDs; missing predictions count as misses and unexpected query IDs are rejected.
An existing split with different contents is left unchanged and reported as an error.

The official COLIEE metric is **micro-averaged F1**: count true positives, false positives and false negatives over all queries, then compute precision, recall and F1 once. All tuning uses the dev split carved out of Train. Test labels are only used for final numbers.
