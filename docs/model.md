# The embedding model (EmbeddingGemma 2)

zemsearch computes embeddings with **EmbeddingGemma 2** served locally by
`llama-server`. This document collects the model facts and the operational
quirks that matter when querying and indexing with it.

Primary sources: the
[model card](https://huggingface.co/google/embeddinggemma-2) and the
[Google AI docs](https://ai.google.dev/gemma/docs/embeddinggemma).

## Overview

- **768-dimensional** output, **mean-pooled**; served over the OpenAI-compatible
  `POST /v1/embeddings` endpoint.
- **8,192-token** context, but text attention uses a **1,024-token sliding
  window** — keep chunks ≤1024 tokens for good long-range signal.
- **MRL (Matryoshka)**: the 768-d vector can be truncated to 512/256/128 and
  re-normalized. Native dimension is always stored; truncation is a query-time
  *view*, so it never requires a reindex.
- **Task-steered**: short instruction prefixes steer the shared backbone per task
  (search, classification …). Code search is an *asymmetric* retrieval task — the
  instruction goes on the query only.

## Prompt prefixes (exact)

These strings are load-bearing: the wrong one degrades ranking **silently** (it
errors neither in the model nor in code). Defined in `src/zemsearch/model/prefixes.py`.

| Side | Format |
| --- | --- |
| Query | `task: code retrieval \| query: {query}` |
| Document | `title: {title or filename} \| text: {code}` |

Notes:

- Code search is **asymmetric**: only the query carries the task instruction; the
  document uses the same `title/… | text/…` template as web search, QA and
  fact-checking.
- Use `title: none` when a document has no natural title.
- The title should be the filename or a `path::symbol`; the body is the content
  (for zemsearch, the chunk text).

## Operational quirks

- **Precision: `BF16` or `FP32` only — never `FP16`.** The activation range
  exceeds fp16's; fp16 returns NaN or silently degraded embeddings without
  raising. `BF16` is the default; Q8_0 also works (see
  [`reports/q8-vs-bf16.md`](../reports/q8-vs-bf16.md)).
- **Loading**: `llama-server` needs `--embeddings --pooling mean --embd-normalize 2`
  (and `--no-mmproj` for the text-only backbone). zemsearch builds this command
  in `model/server.py`.
- **Batch size**: the server's default 512-token physical batch rejects larger
  whole-file inputs; zemsearch passes `--batch-size`/`--ubatch-size` equal to the
  context size. If you see `input ... too large to process`, raise those or lower
  `max_chunk_chars`.
- **Architecture support**: EmbeddingGemma 2 is `gemma-embedding2`
  (`LLM_ARCH_GEMMA_EMBEDDING2`) — an older `llama.cpp` fails with
  `unknown model architecture` (see the README's install section).

## Similarity & storage

- **Normalization is load-bearing.** With unit-length vectors, **cosine,
  dot-product and L2 ranking are identical**, so the store uses the cheapest
  (dot). On *un*normalized vectors, dot ranks by magnitude instead — the failure
  is silent, not an error (see
  [`reports/unnormalized-dot-vs-cosine.md`](../reports/unnormalized-dot-vs-cosine.md)).
- **MRL view.** zemsearch persists the native 768-d vector and, for search, keeps
  the leading `query_dim` dims and **re-normalizes after truncating**:
  `normalize(v[:d])` (`normalize(v)[:d] != normalize(v[:d])`). Default
  `query_dim = 256`, which is near-lossless for text and code and cuts storage
  3×; see [`reports/dim-ranking-256-vs-768.md`](../reports/dim-ranking-256-vs-768.md)
  and [`reports/mrl-truncation-at-query.md`](../reports/mrl-truncation-at-query.md).
- **Queries and documents must share a dimension** — a 256-d query cannot be
  scored against a 768-d corpus.

## How zemsearch uses it

- Documents are formatted with `format_document` and queries with `format_query`
  (`model/prefixes.py`); the query prefix is applied at search time in
  `runtime.py`.
- Vectors are stored native 768-d; the `query_dim` MRL view is applied when
  loading the in-memory snapshot and to the query vector (`view()` in
  `model/client.py`). Changing `query_dim` never triggers a reindex.
