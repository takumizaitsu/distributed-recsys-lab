from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


@dataclass(frozen=True)
class IdMaps:
    """Bidirectional id/index mappings for users and items."""
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_to_index: Dict[int, int]
    item_to_index: Dict[int, int]


def build_id_maps(
    train: pd.DataFrame, user_col: str = "userId", item_col: str = "movieId"
) -> IdMaps:
    """Builds id/index maps from the training data.

    Args:
      train: Training DataFrame.
      user_col: Column name for user id.
      item_col: Column name for item id.

    Returns:
      IdMaps for users/items.
    """
    user_ids = np.array(sorted(train[user_col].unique()), dtype=np.int64)
    item_ids = np.array(sorted(train[item_col].unique()), dtype=np.int64)
    user_to_index = {int(u): i for i, u in enumerate(user_ids)}
    item_to_index = {int(it): i for i, it in enumerate(item_ids)}
    return IdMaps(user_ids, item_ids, user_to_index, item_to_index)


def build_interaction_matrix(
    train: pd.DataFrame,
    maps: IdMaps,
    user_col: str = "userId",
    item_col: str = "movieId",
) -> sparse.csr_matrix:
    """Builds a user-item implicit interaction matrix.

    Each observed (user, item) in `train` becomes an interaction value of 1.

    Args:
      train: Training DataFrame.
      maps: IdMaps built from train.
      user_col: Column name for user id.
      item_col: Column name for item id.

    Returns:
      CSR matrix of shape (num_users, num_items).
    """
    rows = train[user_col].astype(int).map(maps.user_to_index).to_numpy()
    cols = train[item_col].astype(int).map(maps.item_to_index).to_numpy()
    data = np.ones(len(train), dtype=np.float32)
    mat = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(maps.user_ids), len(maps.item_ids)),
        dtype=np.float32,
    )
    mat.sum_duplicates()
    return mat


def fit_svd_mf(
    interactions: sparse.csr_matrix,
    n_factors: int = 64,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fits a simple MF-like model via TruncatedSVD.

    Args:
      interactions: User-item interaction matrix.
      n_factors: Number of latent factors.
      random_state: Random seed for reproducibility.

    Returns:
      (user_factors, item_factors)
        user_factors: shape (num_users, n_factors)
        item_factors: shape (num_items, n_factors)
    """
    svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
    user_factors = svd.fit_transform(interactions).astype(np.float32)
    item_factors = svd.components_.T.astype(np.float32)
    return user_factors, item_factors


def recommend_topk(
    train_matrix: sparse.csr_matrix,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    k: int = 10,
) -> List[List[int]]:
    """Generates top-K item recommendations for each user.

    Already-seen items in `train_matrix` are excluded.

    Args:
      train_matrix: User-item CSR matrix used for masking seen items.
      user_factors: User latent factors.
      item_factors: Item latent factors.
      k: Number of items to recommend.

    Returns:
      A list of length num_users. Each entry is a list of item indices.
    """
    num_items = item_factors.shape[0]
    if k <= 0:
        return [[] for _ in range(train_matrix.shape[0])]
    k = min(k, num_items)

    scores = user_factors @ item_factors.T  # (num_users, num_items)

    train_csr = train_matrix.tocsr()
    for u in range(train_csr.shape[0]):
        start, end = train_csr.indptr[u], train_csr.indptr[u + 1]
        seen = train_csr.indices[start:end]
        scores[u, seen] = -np.inf

    recs: List[List[int]] = []
    for u in range(scores.shape[0]):
        idx = np.argpartition(-scores[u], kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[u, idx])]
        recs.append([int(i) for i in idx])
    return recs
