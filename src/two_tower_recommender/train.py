"""Training loop."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from two_tower_recommender.data import Dataset, Split
from two_tower_recommender.model import TwoTower


@dataclass
class Config:
    """Training settings."""

    dim: int = 64
    hidden: int = 64
    temperature: float = 0.05
    batch: int = 1024
    epochs: int = 6
    lr: float = 3e-3
    weight_decay: float = 1e-6
    logq: bool = True
    seed: int = 0


@dataclass
class History:
    """Mean loss per epoch."""

    losses: list[float] = field(default_factory=list)


def item_log_q(split: Split, n_items: int) -> torch.Tensor:
    """Log of each item's share of training positives (its chance to appear in a batch)."""
    counts = np.bincount(split.train_items, minlength=n_items).astype(np.float64) + 1.0
    return torch.as_tensor(np.log(counts / counts.sum()), dtype=torch.float32)


def train(ds: Dataset, split: Split, cfg: Config) -> tuple[TwoTower, History]:
    """Train the towers on the split's positives; the same seed gives the same model."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = TwoTower(ds, cfg.dim, cfg.hidden, cfg.temperature)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    log_q = item_log_q(split, ds.n_items) if cfg.logq else None
    users = torch.as_tensor(split.train_users)
    items = torch.as_tensor(split.train_items)
    hist = History()
    for _ in range(cfg.epochs):
        model.train()
        order = torch.as_tensor(rng.permutation(len(users)))
        total, batches = 0.0, 0
        for start in range(0, len(order), cfg.batch):
            idx = order[start : start + cfg.batch]
            if len(idx) < 2:
                continue
            loss = model.loss(users[idx], items[idx], log_q)
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            total += loss.item()
            batches += 1
        hist.losses.append(total / max(batches, 1))
    return model, hist
