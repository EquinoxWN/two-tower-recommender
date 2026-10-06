"""Candidate retrieval with FAISS: exact inner product, or an HNSW graph for larger catalogs."""

from __future__ import annotations

import time
from typing import Any

import faiss
import numpy as np


class ItemIndex:
    """Nearest items to a user embedding by inner product (cosine, since vectors are unit length)."""

    def __init__(self, item_emb: np.ndarray, kind: str = "flat", hnsw_m: int = 32) -> None:
        dim = item_emb.shape[1]
        self.index: Any
        if kind == "flat":
            self.index = faiss.IndexFlatIP(dim)
        elif kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = 256
        else:
            raise ValueError(f"unknown index kind {kind!r}")
        self.index.add(np.ascontiguousarray(item_emb, dtype=np.float32))
        self.n = item_emb.shape[0]

    def search(
        self, user_emb: np.ndarray, k: int, seen: list[set[int]] | None = None
    ) -> np.ndarray:
        """Top-k item ids per user, skipping items in that user's seen set."""
        extra = max((len(s) for s in seen), default=0) if seen else 0
        want = min(k + extra, self.n)
        _, ids = self.index.search(np.ascontiguousarray(user_emb, dtype=np.float32), want)
        out = np.full((len(user_emb), k), -1, dtype=np.int64)
        for row, cand in enumerate(ids):
            skip = seen[row] if seen else set()
            kept = [int(c) for c in cand if c >= 0 and int(c) not in skip][:k]
            out[row, : len(kept)] = kept
        return out

    def latency_ms(self, user_emb: np.ndarray, k: int, repeat: int = 1) -> np.ndarray:
        """Wall time of single-user searches in milliseconds (one query at a time, as a server sees them)."""
        times = []
        for _ in range(repeat):
            for row in user_emb:
                start = time.perf_counter()
                self.index.search(row[None, :], k)
                times.append((time.perf_counter() - start) * 1000)
        return np.array(times)
