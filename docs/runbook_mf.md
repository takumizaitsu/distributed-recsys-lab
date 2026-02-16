# Runbook: MF baseline (0001)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

前提：data/ml-latest-small/ratings.csv を置く（dataはcommitしない）

実行コマンド：
python experiments/0001_mf_baseline/run.py

出力：
results/results.json に dataset stats + MF metrics が出る