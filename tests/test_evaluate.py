"""Metrics on hand-checked examples, and the full pipeline on planted data."""

from __future__ import annotations

import math

import numpy as np
import pytest

from two_tower_recommender.data import split_by_time, synthetic
from two_tower_recommender.evaluate import (
    ndcg_at_k,
    popularity_ranking,
    random_ranking,
    recall_at_k,
)
from two_tower_recommender.retrieval import ItemIndex
from two_tower_recommender.train import Config, train


def test_recall_and_ndcg_on_a_worked_example() -> None:
    ranked = np.array([[5, 1, 9, 2], [7, 8, 3, 4]])
    truth = [[1, 2], [4]]
    assert recall_at_k(ranked, truth, 2) == pytest.approx((0.5 + 0.0) / 2)
    assert recall_at_k(ranked, truth, 4) == pytest.approx((1.0 + 1.0) / 2)
    dcg_user1 = 1 / math.log2(3) + 1 / math.log2(5)
    ideal_user1 = 1 + 1 / math.log2(3)
    dcg_user2 = 1 / math.log2(5)
    assert ndcg_at_k(ranked, truth, 4) == pytest.approx((dcg_user1 / ideal_user1 + dcg_user2) / 2)
    assert ndcg_at_k(np.array([[1, 2]]), [[1, 2]], 2) == pytest.approx(1.0)


def test_baselines_skip_seen_items() -> None:
    train_items = np.array([3, 3, 3, 1, 1, 2])
    pop = popularity_ranking(train_items, 5, [set(), {3}], 3)
    assert pop[0].tolist() == [3, 1, 2]
    assert pop[1].tolist() == [1, 2, 0]
    rnd = random_ranking(5, [{0, 1}], 3, seed=0)
    assert not {0, 1} & set(rnd[0].tolist())


def test_on_planted_preferences_the_towers_beat_popularity_by_far() -> None:
    ds = synthetic(n_users=400, n_items=300, seed=5)
    sp = split_by_time(ds)
    model, _ = train(ds, sp, Config(epochs=8, batch=512, seed=0))
    users = np.array(sorted(sp.test))
    truth = [sp.test[int(u)] for u in users]
    seen = [sp.seen[int(u)] for u in users]
    ranked = ItemIndex(model.embed_items(ds.n_items)).search(model.embed_users(users), 20, seen)
    ours = recall_at_k(ranked, truth, 20)
    pop = recall_at_k(popularity_ranking(sp.train_items, ds.n_items, seen, 20), truth, 20)
    assert ours > 3 * pop, (ours, pop)
    assert ours > 0.3
