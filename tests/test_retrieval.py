"""FAISS retrieval matches brute force and respects what users have seen."""

from __future__ import annotations

import numpy as np
import pytest

from two_tower_recommender.retrieval import ItemIndex


def unit(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_exact_search_equals_brute_force() -> None:
    rng = np.random.default_rng(0)
    items, users = unit(rng, 300, 16), unit(rng, 20, 16)
    got = ItemIndex(items, "flat").search(users, 25)
    want = np.argsort(-(users @ items.T), axis=1, kind="stable")[:, :25]
    assert np.array_equal(got, want)


def test_seen_items_are_skipped_and_the_list_is_still_full() -> None:
    rng = np.random.default_rng(1)
    items, users = unit(rng, 100, 8), unit(rng, 3, 8)
    idx = ItemIndex(items, "flat")
    top = idx.search(users, 10)
    seen = [set(top[0, :5].tolist()), set(), set(top[2].tolist())]
    out = idx.search(users, 10, seen)
    for row in range(3):
        assert not seen[row] & set(out[row].tolist())
        assert (out[row] >= 0).all(), "excluded items are replaced by the next best"
    assert out[1].tolist() == top[1].tolist()


def test_hnsw_finds_nearly_all_of_the_exact_neighbours() -> None:
    rng = np.random.default_rng(2)
    items, users = unit(rng, 5000, 32), unit(rng, 50, 32)
    exact = ItemIndex(items, "flat").search(users, 100)
    approx = ItemIndex(items, "hnsw").search(users, 100)
    overlap = np.mean([len(set(a) & set(b)) / 100 for a, b in zip(exact, approx, strict=True)])
    assert overlap > 0.95


def test_asking_for_more_items_than_exist_pads_with_minus_one() -> None:
    rng = np.random.default_rng(3)
    out = ItemIndex(unit(rng, 5, 4), "flat").search(unit(rng, 1, 4), 8)
    assert sorted(out[0, :5].tolist()) == [0, 1, 2, 3, 4]
    assert out[0, 5:].tolist() == [-1, -1, -1]
    with pytest.raises(ValueError, match="unknown index kind"):
        ItemIndex(unit(rng, 5, 4), "ivf")


def test_latency_is_measured_per_query() -> None:
    rng = np.random.default_rng(4)
    idx = ItemIndex(unit(rng, 1000, 16), "flat")
    lat = idx.latency_ms(unit(rng, 10, 16), 50, repeat=2)
    assert lat.shape == (20,)
    assert (lat > 0).all()
