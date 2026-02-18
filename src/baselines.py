from __future__ import annotations

from typing import Dict, List
import pandas as pd


def compute_item_popularity(
    train: pd.DataFrame,
    item_col: str = "movieId",
) -> Dict[int, int]:
    """Counts item popularity in the training set.

    Args:
      train: Training DataFrame.
      item_col: Column name for item id.

    Returns:
      Dict of {item_id: count}.
    """
    counts = train[item_col].value_counts()
    return {int(k): int(v) for k, v in counts.items()}


def recommend_popularity_topk(
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    user_col: str = "userId",
    item_col: str = "movieId",
) -> List[List[int]]:
    """Recommends top-K popular items per user, excluding seen items.

    For each user, excludes items that appear in the user's training history.

    Args:
      train: Training DataFrame.
      test: Test DataFrame (used to define user set for evaluation).
      k: Number of items to recommend.
      user_col: Column name for user id.
      item_col: Column name for item id.

    Returns:
      Per-user recommendations aligned with users in `test` (row order).
    """
    popularity = compute_item_popularity(train, item_col=item_col)
    global_ranked = [item for item, _ in sorted(popularity.items(), key=lambda x: -x[1])]

    # user -> seen items set
    seen_by_user = (
        train.groupby(user_col)[item_col].apply(lambda s: set(int(x) for x in s)).to_dict()
    )

    recs: List[List[int]] = []
    for row in test.itertuples(index=False):
        user_id = int(getattr(row, user_col))
        seen = seen_by_user.get(user_id, set())
        user_recs = []
        for item_id in global_ranked:
            if item_id in seen:
                continue
            user_recs.append(item_id)
            if len(user_recs) >= k:
                break
        recs.append(user_recs)
    return recs
