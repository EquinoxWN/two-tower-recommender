# RFC 0001: two-tower-recommender design

- **Status:** Accepted (M1 implemented)
- **Author:** EquinoxWN
- **Created:** 2026

## Problem

"Recommended for you" has to pick a few dozen items out of a catalog of thousands to millions for
every user, within a request budget of tens of milliseconds. Scoring every item with a rich model
is far too slow, so production systems split the work: a cheap retrieval stage narrows the catalog
to a few hundred candidates, and a heavier ranker orders them. This project builds that pipeline
step by step and measures each step against honest baselines, starting with retrieval.

## Goals

- **M1 (this RFC):**
  - A user tower (user id, gender, age bucket, occupation) and an item tower (item id, genres,
    release decade) map users and items to unit vectors whose dot product predicts interest.
  - Training uses in-batch negatives: each positive pair in a batch of 1,024 is contrasted with
    every other item in the batch, with a temperature of 0.05, the logQ correction for popular
    items, and duplicates of the positive masked out.
  - Item vectors go into a FAISS index that returns 500 candidates per user; exact inner product
    and an HNSW graph are both measured.
  - Offline evaluation on MovieLens 1M with a time-based split: recall@10/50/100/500 and NDCG@10
    against popularity and random baselines, and single-query retrieval latency.
- **M2:** a LightGBM ranker over the candidates with richer features, including real-time ones
  from the streaming feature pipeline, and a diversity re-rank.
- **M3:** ONNX export and serving behind FastAPI within a 50 ms budget; MovieLens 25M; latency per
  stage.

## Non-goals

- Online learning and A/B testing infrastructure.
- Cold start for brand-new items beyond what genre and decade features give the item tower.

## Proposed design

```
ratings.dat ──> positives (rating >= 4) ──> time split per user: last 2 positives = test
                                              │
          user tower: id + gender + age + occupation ──> MLP ──> unit vector u
          item tower: id + genres + decade          ──> MLP ──> unit vector v
                                              │
          loss: softmax over the batch of  u·v / 0.05  -  log q(item)   (duplicates masked)
                                              │
          all item vectors ──> FAISS (exact or HNSW) ──> top 500 unseen items per user
                                              │
          recall@k, NDCG@10 vs popularity and random; latency per single-user query
```

- **Data:** MovieLens 1M (6,040 users, 3,883 movies, 1,000,209 ratings) is downloaded over HTTPS
  once into `.tmp/data`, and refused unless its SHA-256 matches the pinned value (taken from a
  download whose MD5 matched the one GroupLens publishes). It is never committed: the MovieLens
  licence does not allow redistribution.
- **Split:** for every user with at least seven positives, the last two positives by time are
  the test set; everything the user rated before them is "seen" and never recommended, so a model
  is not rewarded for recommending what the user already watched, and no future interaction leaks
  into training.
- **logQ correction:** in-batch negatives are sampled in proportion to popularity, so popular
  items are pushed down far more often than rare ones. Subtracting log q(item), the item's share of
  training positives, from each logit undoes that bias (Yi et al., RecSys 2019).
- **Reproducibility:** one seed fixes initialization and batch order; two runs give identical
  losses and embeddings (tested).

## Alternatives considered

| Option | Why not (yet) |
|---|---|
| Matrix factorization (ALS) | A strong baseline, but it cannot use user or item features; the towers can, which is what makes them work for new users and items later. Worth adding as a baseline in M3. |
| Sampled negatives from the whole catalog | Needs a sampler and more compute per step; in-batch negatives are free, and the logQ correction fixes their bias. |
| Random train/test split | Leaks the future into training and overstates every metric; a time-based split matches how the model is used. |
| HNSW as the default index | At 3,883 items exact search is faster and exact; HNSW pays off at millions of items (M3), and is measured here so the comparison is ready. |

## Measurement plan

- Recall@k and NDCG@10 for the towers, popularity and random, on the same held-out positives.
- Ablation: the same training without the logQ correction.
- Retrieval latency for one user at a time (p50 and p99), exact and HNSW, and HNSW's overlap with
  the exact top 500.

## Milestones

- **M1 (done):** towers, in-batch negatives with logQ, FAISS retrieval, offline evaluation.
- **M2:** LightGBM ranker with streaming features, diversity re-rank.
- **M3:** ONNX and FastAPI within 50 ms, MovieLens 25M, latency per stage.

## Risks and open questions

- MovieLens is ratings of movies people chose to watch, so "positive" means "rated 4 or 5", not
  "would click"; offline recall only approximates online behaviour.
- The test set is two items per user, so recall@10 has high variance per user; the mean over
  6,016 users is stable, and the seed is fixed.
