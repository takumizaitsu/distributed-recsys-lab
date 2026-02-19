from __future__ import annotations

import pandas as pd


def leave_one_out_by_time(
    ratings: pd.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
    time_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits ratings into train/test by time using leave-one-out.

    For each user, the latest interaction (by `time_col`) is used as test,
    and the rest are used as train. Users with only one interaction are
    excluded from the test set (because their train would be empty).

    Args:
      ratings: Ratings DataFrame that contains user/item/rating/timestamp.
      user_col: Column name for user id.
      item_col: Column name for item id.
      time_col: Column name for timestamp.

    Returns:
      (train_df, test_df)

    Raises:
      ValueError: If `time_col` is missing.
    """
    if time_col not in ratings.columns:
        raise ValueError(
            f"Missing column: {time_col}. Need timestamp for leave-one-out."
        )

    r = ratings[[user_col, item_col, "rating", time_col]].dropna()
    r = r.sort_values([user_col, time_col], ascending=True)

    last_idx = r.groupby(user_col, sort=False).tail(1).index
    test = r.loc[last_idx].copy()
    train = r.drop(index=last_idx).copy()

    valid_users = set(train[user_col].unique())
    test = test[test[user_col].isin(valid_users)].copy()

    return train, test


def leave_last_n_by_time(
    ratings: pd.DataFrame,
    n: int = 3,
    user_col: str = "userId",
    item_col: str = "movieId",
    time_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits ratings into train/test by time using leave-last-N.

    For each user, the latest N interactions (by `time_col`) are used as test,
    and the rest are used as train. Users with <= N interactions are excluded
    from the test set (because their train would be empty).

    Args:
      ratings: Ratings DataFrame that contains user/item/rating/timestamp.
      n: Number of latest interactions to use as test per user.
      user_col: Column name for user id.
      item_col: Column name for item id.
      time_col: Column name for timestamp.

    Returns:
      (train_df, test_df)

    Raises:
      ValueError: If `time_col` is missing or n <= 0.
    """
    if time_col not in ratings.columns:
        raise ValueError(
            f"Missing column: {time_col}. Need timestamp for time-based split."
        )
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    r = ratings[[user_col, item_col, "rating", time_col]].dropna()
    r = r.sort_values([user_col, time_col], ascending=True)

    # Latest N per user as test
    test_idx = r.groupby(user_col, sort=False).tail(n).index
    test = r.loc[test_idx].copy()
    train = r.drop(index=test_idx).copy()

    # Exclude users whose train becomes empty (users with <= n interactions)
    valid_users = set(train[user_col].unique())
    test = test[test[user_col].isin(valid_users)].copy()

    return train, test
