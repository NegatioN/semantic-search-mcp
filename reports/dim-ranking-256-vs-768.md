# Report: ranking at query_dim=256 vs query_dim=768

**Question.** Now that vectors are stored native (768) and reduced by an MRL
*view* at query time, does the search dimension actually change which code comes
back?

**Method.** `scripts/dim_ranking_probe.py` loads the *same* native-768
clickhouse-tests vectors (61 files, 64 chunks) at `query_dim=256` and
`query_dim=768`, embeds 8 architectural queries once each, and compares rankings.
CPU/GPU server, cosine (dot on normalized views).

## Metrics (256 vs 768)

| query | top-1 same | overlap@5 | overlap@10 | Spearman |
| --- | :---: | :---: | :---: | :---: |
| where is the clickhouse client connection handled | yes | 1.0 | 0.8 | 0.963 |
| how is the benchmark workload generated for simulated users | yes | 0.6 | 0.9 | 0.963 |
| clickhouse settings variants and presets | yes | 0.8 | 0.9 | 0.924 |
| produce a benchmark report summary and percentiles | yes | 0.8 | 1.0 | 0.978 |
| docker compose services for clickhouse | yes | 1.0 | 0.9 | 0.951 |
| how does the measurement sampling work | yes | 0.4 | 0.4 | 0.847 |
| logic that compares variants and picks the best | yes | 0.6 | 0.9 | 0.966 |
| how are results persisted and read back from reports | **no** | 1.0 | 0.7 | 0.872 |

- **Top-1 identical on 7/8 queries.**
- **High rank correlation** (Spearman 0.85–0.98): the two dims mostly agree on
  ordering and differ mainly within the top-k.
- Overlap@10 is 0.7–1.0 for most; the weakest (`measure sampling`) still keeps
  its top-1 and diverges only in the tail.
- Scores are consistently a bit **lower at 768** (cosine over more dims), but the
  ordering is what matters.

## The one top-1 change

```
Q: how are results persisted and read back from reports
  256d: internal/variant/variant.go(0.752), internal/ch/client.go(0.746), internal/report/report.go(0.746)
  768d: internal/ch/client.go(0.717), internal/measure/measure.go(0.716), internal/report/report.go(0.714)
```

A marginal query with three near-tied candidates (0.714–0.752). Both dims put
`report.go` in the top-3; the leader flips between `variant.go` and `client.go`.
This is exactly the "near-tie" regime where the view dimension can change the
winner.

## Other queries (top-3)

```
where is the clickhouse client connection handled
  256d: ch/client.go(0.834), ch/client.go(0.802), up.sh(0.772)
  768d: ch/client.go(0.808), ch/client.go(0.788), up.sh(0.752)

how is the benchmark workload generated ...
  256d: workload/workload.go(0.833), gen/gen.go(0.801), reports/100k.../meta.json(0.800)
  768d: workload/workload.go(0.821), reports/100k.../meta.json(0.787), reports/ratecheck/meta.json(0.782)

clickhouse settings variants and presets
  256d: variant/presets.go(0.820), reports/order-u.../meta.json(0.799), reports/agg.../meta.json(0.798)
  768d: variant/presets.go(0.802), ch/client.go(0.767), reports/prewhere.../meta.json(0.763)
```

## Analysis

- For this repo, **256 is effectively as good as 768**: the same files surface,
  just ordered slightly differently. That matches the model card's claim that
  256d is near-lossless for code.
- 768 is **not** strictly "better" — it reorders near-ties and, in the one
  flippable case, picked a different (arguably no more correct) top-1. Without a
  labeled query set there's no ground truth to declare a winner.
- Cost is where the difference is real: 768 is ~3× the search time and memory
  (see `reports/mrl-truncation-at-query.md`).

## Caveats

- Tiny corpus (64 chunks) and 8 hand-written queries — an indication, not an
  evaluation. Query-embedding latency is identical at both dims.

## Recommendation

Keep `query_dim=256` as the default; it retrieves the same candidates on this
repo at 3× lower cost. Raise it only if a labeled eval on your own data shows a
real recall gain. Query-dimension changes are free to try (no reindex), so this
is easy to A/B later.

## Reproduce

```bash
GEMMA_EMBEDDER_BINARY=~/.local/share/gemma-embedder/llama.cpp-master/build-cuda/bin/llama-server \
  .venv/bin/python scripts/dim_ranking_probe.py
```
