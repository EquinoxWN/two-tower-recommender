# ADR 0002: Train with in-batch negatives and correct them for popularity (logQ)

- **Status:** Accepted

## Context

A two-tower model learns by contrasting each user's positive item with negatives. Sampling
negatives from the whole catalog costs an extra item-tower pass per negative. Using the other
items in the same batch is free, but those items were drawn from the training positives, so a
popular movie appears as a negative for hundreds of users per epoch while a niche one almost never
does; the model learns to push popular items down.

## Decision

Use in-batch negatives with a sampled softmax over the batch (temperature 0.05), subtract
`log q(item)` from every logit, where `q` is the item's share of training positives, and mask a
batch item that equals the row's own positive (it is not a negative).

## Consequences

- On MovieLens 1M the correction is the difference between beating popularity and losing to it:
  recall@10 is 0.083 with it and 0.036 without (popularity: 0.048), with the same seed, data and
  schedule.
- Tests check the loss against a hand-computed softmax, that the correction subtracts exactly
  `log q`, and that a repeated positive is excluded from its own row's negatives.
- The correction uses training-set frequencies; if the catalog's popularity shifts quickly, `q`
  must be re-estimated, for example with a streaming count.
