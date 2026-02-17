# Runbook: MF Baseline (0001)

This runbook reproduces the Week1 MF baseline experiment using MovieLens (ml-latest-small).

---

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## 2. Data

Download **MovieLens ml-latest-small** and place the file as:

```
data/ml-latest-small/ratings.csv
```

⚠️ The `data/` directory is NOT committed to Git.

Expected file:
```
data/ml-latest-small/ratings.csv
```

---

## 3. Run Experiment

```bash
python experiments/0001_mf_baseline/run.py
```

---

## 4. Output

Results are written to:

```
results/results.json
```

Example `mf` section:

```json
"mf": {
  "method": "truncated_svd",
  "factors": 64,
  "k": 10,
  "seed": 42,
  "train_interactions": 100226,
  "test_rows": 610,
  "eval_users": 587,
  "eval_drop": {
    "missing_user_in_train": 0,
    "missing_item_in_train": 23
  },
  "wall_time_sec": 0.20,
  "recall@10": 0.0783,
  "ndcg@10": 0.0348
}
```

---

## Notes

- Split: Leave-one-out by timestamp (latest interaction per user is test)
- Evaluation: Recall@10 / NDCG@10
- Seen items in train are excluded during recommendation
- Seed is fixed to 42 for reproducibility

This artifact represents the reproducible MF baseline for Week1.
