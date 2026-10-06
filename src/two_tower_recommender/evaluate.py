"""Offline metrics: recall@k and NDCG@k against held-out items, and simple baselines."""

from __future__ import annotations

import math

import numpy as np


def recall_at_k(ranked: np.ndarray, truth: list[list[int]], k: int) -> float:
    """Mean over users of the share of their held-out items found in the top k."""
    vals = [
        len(set(r[:k].tolist()) & set(t)) / len(t) for r, t in zip(ranked, truth, strict=True) if t
    ]
    return float(np.mean(vals)) if vals else 0.0


def ndcg_at_k(ranked: np.ndarray, truth: list[list[int]], k: int) -> float:
    """Mean over users of the discounted gain of hits in the top k, normalized by the best ordering."""
    vals = []
    for r, t in zip(ranked, truth, strict=True):
        if not t:
            continue
        want = set(t)
        dcg = sum(1 / math.log2(pos + 2) for pos, item in enumerate(r[:k].tolist()) if item in want)
        ideal = sum(1 / math.log2(pos + 2) for pos in range(min(len(want), k)))
        vals.append(dcg / ideal)
    return float(np.mean(vals)) if vals else 0.0


def popularity_ranking(
    train_items: np.ndarray, n_items: int, seen: list[set[int]], k: int
) -> np.ndarray:
    """The k most popular training items each user has not seen."""
    order = np.argsort(-np.bincount(train_items, minlength=n_items), kind="stable")
    out = np.full((len(seen), k), -1, dtype=np.int64)
    for row, s in enumerate(seen):
        kept = [int(i) for i in order[: k + len(s)] if int(i) not in s][:k]
        out[row, : len(kept)] = kept
    return out


def random_ranking(n_items: int, seen: list[set[int]], k: int, seed: int = 0) -> np.ndarray:
    """k random unseen items per user."""
    rng = np.random.default_rng(seed)
    out = np.full((len(seen), k), -1, dtype=np.int64)
    for row, s in enumerate(seen):
        cand = [int(i) for i in rng.permutation(n_items) if int(i) not in s][:k]
        out[row, : len(cand)] = cand
    return out
