from __future__ import annotations

from typing import List
import numpy as np


def recall_at_k(recommendations: List[List[int]], truth: List[List[int]], k: int) -> float:
    """Computes Recall@K.

    Args:
      recommendations: Per-user ranked list of recommended item indices.
      truth: Per-user list of relevant item indices.
      k: Cutoff rank.

    Returns:
      Mean Recall@K over users.
    """
    if len(recommendations) != len(truth):
        raise ValueError("recommendations and truth must have the same length.")

    if k <= 0:
        return 0.0

    hits = []
    for rec, rel in zip(recommendations, truth):
        topk = set(rec[:k])
        rel_set = set(rel)
        hits.append(1.0 if (topk & rel_set) else 0.0)

    return float(np.mean(hits)) if hits else 0.0


def ndcg_at_k(recommendations: List[List[int]], truth: List[List[int]], k: int) -> float:
    """Computes NDCG@K.

    For the Day3 setting (usually one relevant item), NDCG is
    1 / log2(rank+1) when the relevant item appears in top-K.

    Args:
      recommendations: Per-user ranked list of recommended item indices.
      truth: Per-user list of relevant item indices.
      k: Cutoff rank.

    Returns:
      Mean NDCG@K over users.
    """
    if len(recommendations) != len(truth):
        raise ValueError("recommendations and truth must have the same length.")

    if k <= 0:
        return 0.0

    scores = []
    for rec, rel in zip(recommendations, truth):
        rel_set = set(rel)
        gain = 0.0
        for rank, item in enumerate(rec[:k], start=1):
            if item in rel_set:
                gain = 1.0 / np.log2(rank + 1)
        scores.append(gain)

    return float(np.mean(scores)) if scores else 0.0
