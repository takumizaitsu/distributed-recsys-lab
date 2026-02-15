from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetStats:
    num_rows: int
    num_users: int
    num_items: int


def load_movielens_ratings(data_dir: str | Path) -> pd.DataFrame:
    """Loads MovieLens ratings data.

    Supports:
      - ratings.csv (e.g., ml-latest-small, ml-20m)
      - u.data (ml-100k)

    Args:
      data_dir: Directory path that contains ratings.csv or u.data.

    Returns:
      A DataFrame with columns: userId, movieId, rating, timestamp.

    Raises:
      FileNotFoundError: If neither ratings.csv nor u.data is found.
      ValueError: If required columns are missing.
    """
    data_dir = Path(data_dir)

    csv_path = data_dir / "ratings.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        required = {"userId", "movieId", "rating", "timestamp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"ratings.csv missing columns: {sorted(missing)}")
        return df[["userId", "movieId", "rating", "timestamp"]]

    udata_path = data_dir / "u.data"
    if udata_path.exists():
        df = pd.read_csv(
            udata_path,
            sep="\t",
            header=None,
            names=["userId", "movieId", "rating", "timestamp"],
        )
        return df[["userId", "movieId", "rating", "timestamp"]]

    raise FileNotFoundError(
        f"ratings.csv or u.data not found under: {data_dir.resolve()}"
    )


def compute_stats(ratings: pd.DataFrame) -> DatasetStats:
    """Computes basic dataset statistics.

    Args:
      ratings: Ratings DataFrame.

    Returns:
      DatasetStats with num_rows/num_users/num_items.
    """
    r = ratings.dropna(subset=["userId", "movieId", "rating"])
    return DatasetStats(
        num_rows=int(len(r)),
        num_users=int(r["userId"].nunique()),
        num_items=int(r["movieId"].nunique()),
    )
