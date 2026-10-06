"""twotower [--synthetic] [--epochs N] [--seed S]: train on MovieLens 1M, index the items with FAISS,
and compare recall@k and NDCG@k with popularity and random baselines, plus retrieval latency."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from two_tower_recommender.data import DataError, download, load_movielens, split_by_time, synthetic
from two_tower_recommender.evaluate import (
    ndcg_at_k,
    popularity_ranking,
    random_ranking,
    recall_at_k,
)
from two_tower_recommender.retrieval import ItemIndex
from two_tower_recommender.train import Config, train

KS = (10, 50, 100, 500)


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and print the report; 2 on a data problem."""
    p = argparse.ArgumentParser(prog="twotower", description=__doc__)
    p.add_argument(
        "--synthetic", action="store_true", help="use the built-in synthetic dataset (no download)"
    )
    p.add_argument("--data", type=Path, default=Path(".tmp/data/ml-1m.zip"))
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-logq", action="store_true", help="train without the popularity (logQ) correction"
    )
    args = p.parse_args(argv)
    t0 = time.perf_counter()
    try:
        ds = synthetic(seed=args.seed) if args.synthetic else load_movielens(download(args.data))
    except (DataError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    sp = split_by_time(ds)
    print(
        f"Data: {ds.n_users:,} users, {ds.n_items:,} items, {len(ds.users):,} ratings; "
        f"{len(sp.train_users):,} training positives, {len(sp.test):,} test users with 2 later positives each "
        f"({time.perf_counter() - t0:.1f} s)"
    )
    t1 = time.perf_counter()
    model, hist = train(ds, sp, Config(epochs=args.epochs, seed=args.seed, logq=not args.no_logq))
    print(
        f"Trained {args.epochs} epochs in {time.perf_counter() - t1:.1f} s; in-batch softmax loss "
        + " -> ".join(f"{x:.2f}" for x in hist.losses)
    )
    test_users = np.array(sorted(sp.test), dtype=np.int64)
    truth = [sp.test[int(u)] for u in test_users]
    seen = [sp.seen.get(int(u), set()) for u in test_users]
    items = model.embed_items(ds.n_items)
    users = model.embed_users(test_users)
    flat, hnsw = ItemIndex(items, "flat"), ItemIndex(items, "hnsw")
    k_max = max(KS)
    rows = {
        "two-tower + FAISS (exact)": flat.search(users, k_max, seen),
        "two-tower + FAISS (HNSW)": hnsw.search(users, k_max, seen),
        "popularity": popularity_ranking(sp.train_items, ds.n_items, seen, k_max),
        "random": random_ranking(ds.n_items, seen, k_max, args.seed),
    }
    head = "| Ranking | " + " | ".join(f"Recall@{k}" for k in KS) + " | NDCG@10 |"
    print()
    print(head)
    print("|" + "---|" * (len(KS) + 2))
    for name, ranked in rows.items():
        cells = " | ".join(f"{recall_at_k(ranked, truth, k):.3f}" for k in KS)
        print(f"| {name} | {cells} | {ndcg_at_k(ranked, truth, 10):.3f} |")
    exact, approx = rows["two-tower + FAISS (exact)"], rows["two-tower + FAISS (HNSW)"]
    overlap = np.mean([len(set(a) & set(b)) / k_max for a, b in zip(exact, approx, strict=True)])
    lat_flat = flat.latency_ms(users[:500], 500)
    lat_hnsw = hnsw.latency_ms(users[:500], 500)
    print()
    print(
        f"Retrieval of 500 candidates for one user over {ds.n_items:,} items (500 single-user queries):"
    )
    for name, lat in (("exact inner product", lat_flat), ("HNSW (M=32, efSearch=256)", lat_hnsw)):
        print(
            f"  {name:28s} p50 {np.percentile(lat, 50):.3f} ms, p99 {np.percentile(lat, 99):.3f} ms"
        )
    print(f"  HNSW returns {overlap:.1%} of the exact top 500")
    return 0


if __name__ == "__main__":
    sys.exit(main())
