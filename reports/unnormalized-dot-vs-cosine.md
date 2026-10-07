# Report: unnormalized embeddings + dot product vs cosine

**Question.** Does the ranking metric matter? Can we just use dot product on the
raw (unnormalized) model output, or does "cosine" actually do work?

**Method.** `scripts/unnormalized_probe.py` runs `llama-server` with
`--embd-normalize -1` (raw, non-unit vectors), embeds the `clickhouse-tests`
corpus (61 files) and a set of queries at `d=256`, then ranks the **same**
vectors two ways:

- **cosine** — L2-normalize every doc and the query, then dot product.
- **dot** — raw dot product on the unnormalized vectors.

The cosine arm reproduces the production pipeline exactly (server-normalized
768 → truncate → renormalize), so any difference is purely the normalization
effect.

## Vector magnitude spread

```
raw 256-d norms: min=131.0  median=168.1  max=583.6
```

Raw vectors differ in magnitude by ~4.5×. Dot product is monotone in that
magnitude, so it partially ranks documents by "how big is this vector" rather
than by similarity.

## Results

```
query                                              top1 same  ov@5  spearman
where is the clickhouse client connection handled       no     0.0    0.291
how is the benchmark workload generated ...             no     0.0    0.019
clickhouse settings variants and presets                no     0.0    0.238
produce a benchmark report summary and percentiles      no     0.4    0.375
docker compose services for clickhouse                  no     0.6    0.289
how does the measurement sampling work                  no     0.0   -0.421
logic that compares variants and picks the best         no     0.0    0.170
```

- **Top-1 changed on 7/7 queries.** With raw dot product, the top hit is
  `.gitignore` for *every* query — it has the largest vector norm, so it wins on
  magnitude regardless of the question.
- **Mean overlap@5 = 0.14**; Spearman correlation between the two score vectors
  drops as low as **−0.42** (i.e. they can even anti-correlate).
- The **cosine** arm returns the semantic answers we expect and matches the
  production index: `internal/ch/client.go`, `internal/workload/workload.go`,
  `internal/variant/presets.go`, `internal/report/report.go`, `docker-compose.yml`,
  `internal/measure/measure.go`.

## Analysis

- The model is trained/configured for **cosine** (`config_sentence_transformers.json`
  declares `"similarity_fn_name": "cosine"`), and the model card warns to
  L2-normalize before cosine similarity.
- Dot product on unnormalized vectors is **not** equivalent to cosine; it is
  cosine scaled by `‖a‖·‖b‖`, so documents with large norms are systematically
  favored. Here that degenerates to "always return the biggest-norm document."
- Because raw norms vary by ~4.5× on real code, the failure is not subtle — it is
  total. Normalization is not a nicety; it is load-bearing.

## Conclusion

Keep normalizing. Any of {cosine, normalized dot, L2 distance} ranks identically
once vectors are unit-length — which is why the production pipeline normalizes
both server-side (`--embd-normalize 2`) and after MRL truncation on the client,
then searches with a dot product. "Use dot product" is only safe **because** the
vectors are normalized; on raw embeddings it silently ranks by magnitude.

## Reproduce

```bash
GEMMA_EMBEDDER_BINARY=~/.local/share/gemma-embedder/llama.cpp-master/build-cuda/bin/llama-server \
  .venv/bin/python scripts/unnormalized_probe.py
```
