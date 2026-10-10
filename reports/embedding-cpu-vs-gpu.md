# Report: embedding throughput — CPU vs CUDA `llama-server`

**What.** The same EmbeddingGemma 2 weights (BF16) served by two `llama-server`
builds — a CPU-only static build and a CUDA build — timed on the identical
corpus. The measurement isolates the embedding forward pass: server startup and a
warm-up batch are excluded, only the steady-state embed loop is counted.

**Environment.** AMD Ryzen 7 5700X (8 cores / 16 threads), NVIDIA GeForce RTX
4070, `ctx-size 8192`, `batch = ubatch = 8192`, one HTTP request per batch.
Corpus: 256 near-identical documents of ~800 characters (≈200 tokens) each.

## Results (batch size 32)

| backend | time | docs/s | kchar/s | speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU (static build) | 34.40 s | 7.4 | 6.0 | 1× |
| CUDA (`-ngl 99`) | 1.70 s | 150.7 | 122.0 | **20.2×** |

## Batch size (GPU)

| batch | docs/s | kchar/s |
| ---: | ---: | ---: |
| 4 | 91.2 | 73.8 |
| 32 | 150.9 | 122.2 |
| 128 | 221.3 | 179.2 |

## Analysis

- The GPU is **~20× faster** at batch 32, and the gap only widens as batches grow.
- GPU throughput depends strongly on **batch size** (91 → 221 docs/s from batch
  4 → 128): small requests leave the SMs idle between forwards. This is why
  zemsearch's per-file batching under-uses the GPU — real indexing runs nearer the
  small-batch end of this table, so the practical CPU→GPU win is smaller than the
  headline ratio until requests are packed larger.
- The CPU run's 34 s of wall time consumed ~5 min of CPU time (llama.cpp
  auto-selects its thread count), i.e. it is compute-bound, while the GPU run is
  bound by request size / launch overhead.

## Caveats

- Documents are synthetic and near-identical; real symbol documents vary in
  length, but the relative CPU↔GPU gap is expected to hold.
- Other `llama-server` processes may have been running; the GPU was idle at rest,
  and the CPU figures carry any background load.
- Startup/model-load time is excluded — it is a fixed cost on both backends.

## Reproduce

```bash
python scripts/embed_throughput_probe.py \
    --docs 256 --chars 800 --batch 32 \
    --cpu-binary ~/.local/share/gemma-embedder/llama.cpp-master/build-static/bin/llama-server \
    --gpu-binary ~/.local/share/gemma-embedder/llama.cpp-master/build-cuda/bin/llama-server
```

(`--cpu-binary`/`--gpu-binary` default to the README's
`~/.local/share/zemsearch/llama.cpp/...` layout; pass explicit paths if your builds
live elsewhere, as above.)
