"""The two towers and the in-batch sampled-softmax loss with logQ correction."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from two_tower_recommender.data import AGE_BUCKETS, Dataset


class UserTower(nn.Module):
    """User id plus gender, age bucket and occupation, through an MLP to a unit vector."""

    g: torch.Tensor
    a: torch.Tensor
    o: torch.Tensor

    def __init__(self, ds: Dataset, dim: int, hidden: int) -> None:
        super().__init__()
        self.id = nn.Embedding(ds.n_users, hidden)
        self.gender = nn.Embedding(2, 8)
        self.age = nn.Embedding(len(AGE_BUCKETS), 8)
        self.occupation = nn.Embedding(21, 8)
        self.mlp = nn.Sequential(nn.Linear(hidden + 24, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.register_buffer("g", torch.as_tensor(ds.user_gender))
        self.register_buffer("a", torch.as_tensor(ds.user_age))
        self.register_buffer("o", torch.as_tensor(ds.user_occupation))

    def forward(self, users: torch.Tensor) -> torch.Tensor:
        """Unit-length embeddings for user indices."""
        x = torch.cat(
            [
                self.id(users),
                self.gender(self.g[users]),
                self.age(self.a[users]),
                self.occupation(self.o[users]),
            ],
            dim=1,
        )
        return nn.functional.normalize(self.mlp(x), dim=1)


class ItemTower(nn.Module):
    """Item id plus genres and release decade, through an MLP to a unit vector."""

    gen: torch.Tensor
    yr: torch.Tensor

    def __init__(self, ds: Dataset, dim: int, hidden: int) -> None:
        super().__init__()
        self.id = nn.Embedding(ds.n_items, hidden)
        self.genre = nn.Linear(ds.item_genres.shape[1], 16, bias=False)
        self.year = nn.Embedding(10, 8)
        self.mlp = nn.Sequential(nn.Linear(hidden + 24, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.register_buffer("gen", torch.as_tensor(ds.item_genres))
        self.register_buffer("yr", torch.as_tensor(ds.item_year))

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """Unit-length embeddings for item indices."""
        x = torch.cat(
            [self.id(items), self.genre(self.gen[items]), self.year(self.yr[items])], dim=1
        )
        return nn.functional.normalize(self.mlp(x), dim=1)


class TwoTower(nn.Module):
    """A user tower and an item tower whose dot product scores interest."""

    def __init__(
        self, ds: Dataset, dim: int = 64, hidden: int = 64, temperature: float = 0.05
    ) -> None:
        super().__init__()
        self.user = UserTower(ds, dim, hidden)
        self.item = ItemTower(ds, dim, hidden)
        self.temperature = temperature

    def loss(
        self, users: torch.Tensor, items: torch.Tensor, log_q: torch.Tensor | None = None
    ) -> torch.Tensor:
        """In-batch sampled softmax: each user's positive item against every other item in the batch.

        log_q holds log sampling probabilities of the items: popular items appear as negatives more
        often, so subtracting log q (Yi et al., 2019) keeps the model from simply learning to push
        popular items down. A batch item equal to the row's positive is not a negative and is masked.
        """
        u, v = self.user(users), self.item(items)
        logits = u @ v.T / self.temperature
        if log_q is not None:
            logits = logits - log_q[items][None, :]
        same = items[None, :] == items[:, None]
        same.fill_diagonal_(False)
        logits = logits.masked_fill(same, float("-inf"))
        return nn.functional.cross_entropy(logits, torch.arange(len(users)))

    @torch.no_grad()
    def embed_items(self, n_items: int) -> np.ndarray:
        """Every item's embedding, float32 for FAISS."""
        self.eval()
        out: np.ndarray = self.item(torch.arange(n_items)).numpy().astype(np.float32)
        return out

    @torch.no_grad()
    def embed_users(self, users: np.ndarray) -> np.ndarray:
        """Embeddings for the given users."""
        self.eval()
        out: np.ndarray = self.user(torch.as_tensor(users)).numpy().astype(np.float32)
        return out
