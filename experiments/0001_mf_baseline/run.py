import json
from pathlib import Path
import time

def main():
    # placeholder params (seed固定の文化を最初から入れる)
    params = {
        "seed": 42,
        "k": 10,
        "epochs": 5,
        "factors": 32,
        "lr": 0.01,
        "reg": 0.02,
        "dataset": "movielens-100k",
    }

    started = time.time()

    # TODO: load data, train MF, evaluate
    metrics = {
        "recall@10": None,
        "ndcg@10": None,
    }

    out = {
        "params": params,
        "metrics": metrics,
        "runtime_sec": round(time.time() - started, 3),
    }

    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
