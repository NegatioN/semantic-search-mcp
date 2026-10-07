# Report: Q8_0 vs BF16 (quantized vs native precision)

**Question.** We were serving `embeddinggemma-2-Q8_0` (8-bit). Does moving to the
non-quantized **BF16** weights (the model's native precision, what the model card
benchmarks use) change embeddings or rankings?

**Method.** `scripts/quant_ranking_probe.py` serves the same repo twice
(`Q8_0`, then `BF16`, both with `--no-mmproj` for text-only), embeds the
clickhouse-tests corpus (61 files) and 8 queries with each, and compares.

## Vector agreement (native 768)

```
doc-vector cosine (Q8_0 vs BF16): min=0.99959  mean=0.99979  max=0.99987
max |Δelement| = 9.4e-3
```

Quantization perturbs individual elements by <0.01 and leaves directions
~0.9998 cosine apart.

## Ranking agreement (`query_dim=256`)

| query | top-1 same | overlap@5 | overlap@10 | Spearman |
| --- | :---: | :---: | :---: | :---: |
| where is the clickhouse client connection handled | yes | 1.0 | 1.0 | 0.998 |
| how is the benchmark workload generated for simulated users | yes | 1.0 | 0.9 | 0.999 |
| clickhouse settings variants and presets | yes | 1.0 | 0.9 | 0.998 |
| produce a benchmark report summary and percentiles | yes | 0.8 | 1.0 | 0.997 |
| docker compose services for clickhouse | yes | 1.0 | 1.0 | 0.999 |
| how does the measurement sampling work | yes | 1.0 | 1.0 | 0.998 |
| logic that compares variants and picks the best | yes | 1.0 | 1.0 | 0.996 |
| how are results persisted and read back from reports | yes | 0.8 | 1.0 | 0.995 |

**Top-1 identical on 8/8**; Spearman ≥ 0.995; only the tail of the top-5 shifts.

## Conclusion

- On this repo, **Q8_0 and BF16 are effectively interchangeable** — same top-1,
  near-identical order. Q8_0 was not hurting retrieval.
- We still switched the default to **BF16**: it's the model's native precision
  (reproduces the card's numbers), it costs only ~250 MB more for a 270M text
  model, and on CUDA it uses native bf16 kernels. Q8_0 remains a fine option for
  space/speed-constrained CPU deployments.
- Switching is safe: `model_repo` in the index `meta` includes the quant tag, so
  changing it triggers a one-time forced reindex (no mixed vectors).

## Text-only

The server is started with `--no-mmproj`, so it fetches and loads **only**
`embeddinggemma-2-BF16.gguf` (the 270M text model) and never the vision
projector. Verified: `mmproj-embeddinggemma-2-BF16.gguf` was **not** downloaded.

## Reproduce

```bash
GEMMA_EMBEDDER_BINARY=~/.local/share/gemma-embedder/llama.cpp-master/build-cuda/bin/llama-server \
  .venv/bin/python scripts/quant_ranking_probe.py
```
