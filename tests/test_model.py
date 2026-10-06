"""The towers and the in-batch loss."""

from __future__ import annotations

import math

import numpy as np
import torch

from two_tower_recommender.data import split_by_time, synthetic
from two_tower_recommender.model import TwoTower
from two_tower_recommender.train import Config, item_log_q, train


def test_towers_give_unit_vectors_of_the_chosen_size() -> None:
    ds = synthetic(n_users=20, n_items=30, seed=0)
    m = TwoTower(ds, dim=16, hidden=8)
    u = m.user(torch.arange(5))
    v = m.item(torch.arange(7))
    assert u.shape == (5, 16)
    assert v.shape == (7, 16)
    assert torch.allclose(u.norm(dim=1), torch.ones(5), atol=1e-5)
    assert torch.allclose(v.norm(dim=1), torch.ones(7), atol=1e-5)


def test_the_loss_is_softmax_over_the_other_items_in_the_batch() -> None:
    ds = synthetic(n_users=20, n_items=30, seed=0)
    torch.manual_seed(0)
    m = TwoTower(ds, dim=8, hidden=8, temperature=0.1)
    users, items = torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])
    u, v = m.user(users), m.item(items)
    logits = (u @ v.T / 0.1).detach().numpy()
    manual = np.mean([-(logits[i, i] - math.log(np.exp(logits[i]).sum())) for i in range(3)])
    assert abs(m.loss(users, items).item() - manual) < 1e-4


def test_logq_correction_subtracts_each_items_log_probability() -> None:
    ds = synthetic(n_users=20, n_items=30, seed=0)
    torch.manual_seed(0)
    m = TwoTower(ds, dim=8, hidden=8, temperature=0.1)
    users, items = torch.tensor([0, 1]), torch.tensor([3, 4])
    log_q = torch.zeros(30)
    log_q[4] = math.log(0.5)
    u, v = m.user(users), m.item(items)
    logits = (u @ v.T / 0.1).detach().numpy() - np.array([0.0, math.log(0.5)])
    manual = np.mean([-(logits[i, i] - math.log(np.exp(logits[i]).sum())) for i in range(2)])
    assert abs(m.loss(users, items, log_q).item() - manual) < 1e-4


def test_a_repeated_positive_is_not_used_as_a_negative() -> None:
    ds = synthetic(n_users=20, n_items=30, seed=0)
    torch.manual_seed(0)
    m = TwoTower(ds, dim=8, hidden=8, temperature=0.1)
    users, items = torch.tensor([0, 1, 2]), torch.tensor([3, 3, 5])
    u, v = m.user(users), m.item(items)
    logits = (u @ v.T / 0.1).detach().numpy()
    # Row 0 must not see column 1 (the same item) as a negative, and vice versa.
    rows = [[0, 2], [1, 2], [0, 1, 2]]
    manual = np.mean(
        [-(logits[i, i] - math.log(np.exp(logits[i, rows[i]]).sum())) for i in range(3)]
    )
    loss = m.loss(users, items).item()
    assert math.isfinite(loss)
    assert abs(loss - manual) < 1e-4


def test_item_log_q_is_a_log_distribution() -> None:
    ds = synthetic(n_users=40, n_items=50, seed=0)
    sp = split_by_time(ds)
    log_q = item_log_q(sp, ds.n_items)
    assert abs(float(log_q.exp().sum()) - 1) < 1e-5
    counts = np.bincount(sp.train_items, minlength=ds.n_items)
    assert int(log_q.argmax()) == int(counts.argmax())


def test_training_lowers_the_loss_and_is_reproducible() -> None:
    ds = synthetic(n_users=150, n_items=120, seed=1)
    sp = split_by_time(ds)
    cfg = Config(epochs=4, batch=256, seed=7)
    m1, h1 = train(ds, sp, cfg)
    m2, h2 = train(ds, sp, cfg)
    assert h1.losses[-1] < h1.losses[0]
    assert h1.losses == h2.losses
    assert np.allclose(m1.embed_items(ds.n_items), m2.embed_items(ds.n_items))
