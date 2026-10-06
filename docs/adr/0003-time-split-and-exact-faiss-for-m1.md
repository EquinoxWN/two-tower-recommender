# ADR 0003: Evaluate on each user's latest positives, and retrieve with exact FAISS search for now

- **Status:** Accepted

## Context

Offline numbers are only useful if they resemble how the model is used: it is trained on the past
and asked about the future. A random split puts later ratings in training and earlier ones in the
test set, which inflates every metric. For retrieval, an approximate index (HNSW) is what large
systems use, but approximation only pays off when exact search is too slow.

## Decision

- Hold out each user's last two positives by time; everything rated before them is training data
  (if positive) and is excluded from that user's recommendations (whatever the rating).
- Compare against popularity (most popular unseen items) and random, on exactly the same users,
  held-out items and exclusions.
- Serve candidates from an exact inner-product index (`IndexFlatIP`) in M1, and measure an HNSW
  index (M=32, efSearch=256) next to it.

## Consequences

- Popularity is a strong baseline on MovieLens (recall@10 of 0.048 against 0.002 for random), so
  beating it means the towers learned something personal.
- With 3,883 items, exact search returns 500 candidates in about a quarter of a millisecond (p50)
  and HNSW is slower while finding 98.1% of the same items. The HNSW path is ready for MovieLens
  25M and larger catalogs (M3), where exact search would grow linearly.
