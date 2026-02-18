from __future__ import annotations
from pathlib import Path

import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
# NOTE: Temporary path hack for local experiments. Consider packaging `src/`
# (e.g., editable install) to avoid sys.path manipulation.
sys.path.insert(0, str(ROOT))

from src.data import load_movielens_ratings, compute_stats  # noqa: E402
from src.split import leave_one_out_by_time  # noqa: E402
from src.mf import (  # noqa: E402
    build_id_maps,
    build_interaction_matrix,
    fit_svd_mf,
    recommend_topk,
)
from src.metrics import recall_at_k, ndcg_at_k  # noqa: E402
from src.baselines import recommend_popularity_topk  # noqa: E402



DATASET_NAME = "ml-latest-small"
SEED = 42
N_FACTORS = 64
TOPK = 10


def main() -> None:
    start = time.perf_counter()

    data_dir = ROOT / "data" / DATASET_NAME
    ratings = load_movielens_ratings(data_dir)

    train, test = leave_one_out_by_time(ratings)

    maps = build_id_maps(train)
    train_matrix = build_interaction_matrix(train, maps)
    train_interactions = int(train_matrix.nnz)

    user_factors, item_factors = fit_svd_mf(
        train_matrix, n_factors=N_FACTORS, random_state=SEED
    )
    recommendations = recommend_topk(train_matrix, user_factors, item_factors, k=TOPK)

    pop_recs_full = recommend_popularity_topk(train, test, k=TOPK)

    truth = []
    filtered_mf_recs = []
    filtered_pop_recs = []

    missing_user_in_train = 0
    missing_item_in_train = 0

    for i, row in enumerate(test.itertuples(index=False)):
        user_id = int(row.userId)
        item_id = int(row.movieId)

        if user_id not in maps.user_to_index:
            missing_user_in_train += 1
            continue
        if item_id not in maps.item_to_index:
            missing_item_in_train += 1
            continue

        uidx = maps.user_to_index[user_id]
        truth.append([maps.item_to_index[item_id]])

        filtered_mf_recs.append(recommendations[uidx])
        # popularityはraw movieIdなので、truthと同じ空間にするためmovieId→index変換する
        filtered_pop_recs.append([maps.item_to_index[x] for x in pop_recs_full[i] if x in maps.item_to_index])

    eval_users = len(truth)
    test_rows = len(test)
    
    mf_r10 = recall_at_k(filtered_mf_recs, truth, k=TOPK)
    mf_n10 = ndcg_at_k(filtered_mf_recs, truth, k=TOPK)

    pop_r10 = recall_at_k(filtered_pop_recs, truth, k=TOPK)
    pop_n10 = ndcg_at_k(filtered_pop_recs, truth, k=TOPK)


    wall_time_sec = float(time.perf_counter() - start)

    stats = compute_stats(ratings)
    results = {
        "dataset": DATASET_NAME,
        "num_rows": stats.num_rows,
        "num_users": stats.num_users,
        "num_items": stats.num_items,
        "baselines": {
            "popularity": {"k": TOPK, "recall@10": pop_r10, "ndcg@10": pop_n10}
            },
        "mf": {
            "method": "truncated_svd",
            "factors": N_FACTORS,
            "k": TOPK,
            "seed": SEED,
            "train_interactions": train_interactions,
            "test_rows": test_rows,
            "eval_users": eval_users,
            "eval_drop": {
                "missing_user_in_train": missing_user_in_train,
                "missing_item_in_train": missing_item_in_train,
            },
            "wall_time_sec": wall_time_sec,
            "recall@10": mf_r10,
            "ndcg@10": mf_n10,
        },
    }

    out_path = ROOT / "results" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Wrote: {out_path}")
    print(results["mf"])



if __name__ == "__main__":
    main()
    
