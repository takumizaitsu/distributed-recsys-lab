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
    data_dir = Path(data_dir)

    csv_path = data_dir / "ratings.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df[["userId", "movieId", "rating"]]

    udata_path = data_dir / "u.data"
    if udata_path.exists():
        df = pd.read_csv(
            udata_path,
            sep="\t",
            header=None,
            names=["userId", "movieId", "rating", "timestamp"],
        )
        return df[["userId", "movieId", "rating"]]

    raise FileNotFoundError(f"ratings.csv or u.data not found under: {data_dir.resolve()}")


def compute_stats(ratings: pd.DataFrame) -> DatasetStats:
    r = ratings.dropna(subset=["userId", "movieId", "rating"])
    return DatasetStats(
        num_rows=int(len(r)),
        num_users=int(r["userId"].nunique()),
        num_items=int(r["movieId"].nunique()),
    )
