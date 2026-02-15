from __future__ import annotations

import json
from pathlib import Path
import sys

# src を import できるようにする（後でパッケージ化して綺麗にしてOK）
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import load_movielens_ratings, compute_stats  # noqa: E402


def main() -> None:
    # ここはあなたのデータ配置に合わせて変える
    # 例: data/ml-latest-small/ratings.csv
    data_dir = ROOT / "data" / "ml-latest-small"

    ratings = load_movielens_ratings(data_dir)
    stats = compute_stats(ratings)

    results = {
        "num_rows": stats.num_rows,
        "num_users": stats.num_users,
        "num_items": stats.num_items,
    }

    out_path = ROOT / "results" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}")
    print(results)


if __name__ == "__main__":
    main()
