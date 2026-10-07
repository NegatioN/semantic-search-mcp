# gemma-embedder — Implementation Guide

Local semantic code search powered by **EmbeddingGemma 2**, run natively via
`llama.cpp`, exposed as an MCP server with periodic re-indexing.

This guide is the outcome of reading the goal in `README.md`, researching
EmbeddingGemma 2 (Google blog, Hugging Face model/GGUF cards, llama.cpp server
docs, Qdrant MRL/quantization post), and producing an implementation plan.

---

## Recommended stack

```
llama.cpp llama-server + embeddinggemma-2-GGUF (BF16, ~558MB)
   → POST /v1/embeddings  (L2-normalized, 768d)
Python 3.12/3.13 (uv) + FastMCP (stdio)
   ├─ watchdog + periodic mtime/hash reconciliation
   ├─ tree-sitter symbol chunking (≤1024 tokens)
   ├─ vector store: numpy brute-force → sqlite-vec later
   └─ SQLite (WAL): files / chunks / embeddings / meta
```

---

## Answers to the three questions

### 1. How to run EmbeddingGemma 2 natively (no Ollama)

Use `llama-server` with the GGUF repo — it auto-downloads and caches:

```bash
llama serve -hf ggml-org/embeddinggemma-2-GGUF:BF16 \
  --embeddings --pooling mean --embd-normalize 2 \
  --ctx-size 8192 --host 127.0.0.1 --port 8080
# add --n-gpu-layers 99 if GPU; CPU is fine (270M text backbone)
```

Key gotchas:

- **Never fp16** — the activation range exceeds fp16 and the model returns NaN
  or silently degraded embeddings. Use BF16/Q8_0/fp32.
- The model is **mean-pooled, 768-dimensional**.
- The 8K context uses a **1024-token sliding attention window**, so keep chunks
  ≤1024 tokens for good long-range signal.
- Have the MCP server manage the `llama-server` subprocess, but allow pointing at
  an external `server_url` (remote/shared server).
- Health check `GET /health`; warm up by verifying dim=768, unit norm, and
  determinism before serving.

### 2. How to build the re-indexing MCP server

Python **FastMCP**. Three reindex triggers — debounced `watchdog`, periodic
reconciliation, and an on-demand tool — all funnel into **one asyncio
queue/worker**, so mutations serialize with no lock contention.

Search reads an immutable in-memory snapshot, swapped atomically after each DB
commit. Exposed tools:

- `semantic_search(query, path?, k?, min_score?, granularity?)`
- `get_context(path, start_line, end_line?)`
- `reindex(path?, force?)`
- `index_status()`

Each search hit returns **path + line range + score + snippet**, so a coding
agent can consume it directly.

### 3. Should we embed symbols instead of files?

Yes — but **in addition to** files, not instead. Start whole-file (Phase 1),
then add tree-sitter symbol/function/class chunks
(`title: path::symbol` + line ranges) in Phase 3 behind a `granularity` filter.
Whole-file chunks cover non-parseable code and provide surrounding context.

---

## Embedding prefixes (critical, exact)

EmbeddingGemma 2 is trained with task-instruction prefixes. Use the exact strings:

- **Document:** `title: {path or path::symbol} | text: {code}`
- **Query:** `task: code retrieval | query: {query}`

---

## Milestones

- **P0 (prototype first):** one file → embed → 5 queries through the real
  `llama-server`; assert dim=768, unit norm, and correct chunk ranks #1.
- **P1:** whole-file index + numpy brute force + `semantic_search`/`index_status`
  over stdio.
- **P2:** watcher + incremental reindex + SQLite persistence.
- **P3:** tree-sitter symbol chunking + `get_context`.
- **P4:** MRL dim=256, sqlite-vec ANN, optional 768 rescoring.

---

## Similarity / storage notes

- All vectors are L2-normalized ⇒ **cosine == dot == L2 ranking** (use dot, it's
  cheapest). Use the server's `--embd-normalize 2`. **Normalization is
  load-bearing**: dot product on raw, unnormalized vectors ranks by magnitude
  (see `reports/unnormalized-dot-vs-cosine.md`).
- Vectors are **stored native (768)**; a **query dimension** (`query_dim`,
  default 256) is an MRL *view* `normalize(v[:d])` applied at load/query time.
  **256d is near-lossless for code**; changing `query_dim` never reindexes.
  **Re-normalize after truncation.**
- Exact numpy brute force is plenty for typical local repos (a few hundred
  thousand chunks); only reach for ANN (sqlite-vec/HNSW) when measured to be
  needed.
- Store the model id + native/storage dim + normalization mode in an index
  `meta` table; a **model/normalization** mismatch forces a full reindex
  (embedding spaces are incompatible). The query dim is a view, not a fact.

---

## Project layout (target)

```
gemma-embedder/
├── README.md
├── GUIDE.md
├── pyproject.toml
├── gemma-embedder.toml
├── src/gemma_embedder/
│   ├── cli.py
│   ├── config.py
│   ├── model/
│   │   ├── client.py          # EmbeddingClient (HTTP, batching, retries)
│   │   ├── server.py          # ServerManager (spawn/health/warmup/shutdown)
│   │   └── prefixes.py        # query/doc prompt formatting (pure)
│   ├── chunking/
│   │   ├── base.py
│   │   ├── file.py
│   │   ├── symbols.py         # tree-sitter
│   │   └── languages.py
│   ├── index/
│   │   ├── walker.py
│   │   ├── store.py
│   │   ├── vectors.py
│   │   ├── snapshot.py
│   │   ├── indexer.py
│   │   └── watcher.py
│   ├── mcp/
│   │   ├── server.py
│   │   ├── tools.py
│   │   └── schemas.py
│   └── eval/harness.py
├── scripts/probe_model.py     # Phase 0 spike
└── tests/
```

Entrypoints: `gemma-embedder serve` (MCP), `index`, `search`, `status`, `doctor`.

---

## Phase 0 results (verified)

`scripts/probe_model.py` passed end-to-end: 768-dim, unit-norm, deterministic
embeddings, and correct top-1 file for all six semantic queries.

```bash
# run the probe (spawns llama-server on a free port, stops it afterwards)
python3 scripts/probe_model.py --binary /path/to/llama-server --root .
```

### Important: the prebuilt `llama` binary is too old

The llama.app installer (`curl -LsSf https://llama.app/install.sh | sh`) currently
ships **b11429**, which fails with:

```
error loading model: unknown model architecture: 'gemma-embedding2'
```

Upstream `llama.cpp` master supports it (`LLM_ARCH_GEMMA_EMBEDDING2`). Build from
source until the prebuilt bucket catches up:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/opencode/llama-master
cmake -S /tmp/opencode/llama-master -B /tmp/opencode/llama-master/build \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build /tmp/opencode/llama-master/build --target llama-server -j "$(nproc)"
# binary: /tmp/opencode/llama-master/build/bin/llama-server
```

Add `-DGGML_CUDA=ON` for a GPU build (`nvcc` present here); the CPU build is
plenty for the 270M text backbone.

### Batch size matters for whole-file embeddings

`llama-server` defaults to a 512-token physical batch, which rejects larger
files. `build_command()` now passes `--batch-size`/`--ubatch-size` equal to
`--ctx-size` (8192). Real symbol-level chunking (Phase 3) keeps chunks ≤1024
tokens anyway.

---

## Phase 1 status (implemented)

Whole-file index + exact NumPy search, exposed through both a CLI and an MCP
server over stdio.

### Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

### CLI

```bash
BIN=/path/to/llama-server          # source-built (see above)
.venv/bin/gemma-embedder index  --root . --binary "$BIN"
.venv/bin/gemma-embedder search "how are embeddings stored and loaded" -k 5 --root . --binary "$BIN"
.venv/bin/gemma-embedder status --root .
.venv/bin/gemma-embedder serve  --root . --binary "$BIN"   # MCP over stdio
```

`--binary` can be omitted by setting `binary` in `gemma-embedder.toml` or the
`GEMMA_EMBEDDER_BINARY` env var. Use `--no-spawn` to target an external server.

### MCP configuration

```json
{
  "mcp": {
    "gemma-embedder": {
      "type": "local",
      "command": [
        "/abs/path/.venv/bin/gemma-embedder", "serve",
        "--root", "/abs/path",
        "--binary", "/abs/path/to/llama-server"
      ],
      "enabled": true
    }
  }
}
```

Tools exposed: `semantic_search`, `get_context`, `reindex`, `index_status`.

### Architecture (Phase 1)

- `index/walker.py` — gitignore-aware discovery (pathspec), binary/size pruning.
- `chunking/file.py` — whole-file chunks; line-aligned sliding window
  (6000 chars, 600 overlap) for oversized files.
- `index/store.py` — SQLite (WAL) with `files` / `chunks` / `embeddings` / `meta`;
  float32 vector blobs; sha256 per file for incremental updates.
- `index/vectors.py` — immutable NumPy snapshot, exact dot-product search
  (vectors stored native 768, L2-normalized, viewed at `query_dim=256` via MRL).
- `runtime.py` — shared wiring used by CLI and MCP.
- `mcp/server.py` — official MCP SDK v2 `MCPServer`, stdio transport.

### Verified

- Unit tests: `30 passed`.
- End-to-end: 35 files → 40 chunks; incremental re-index 0.02s (0 changed).
- MCP stdio round-trip: all four tools return correct results; `reindex`
  detects changed files after edits.

### Known limitation (Phase 3 target)

Very short files (e.g. package `__init__.py` docstrings) can rank highly for
project-name queries because a whole-file embedding is dominated by the
`title:` path. Symbol-level chunking plus a minimum-chunk-size filter will
address this.

---

## Phase 2 status (implemented)

Live re-indexing while the MCP server runs.

- `index/watcher.py` — `ReindexScheduler`: a watchdog observer feeds a queue; a
  single worker thread waits for a debounce quiet period, then runs one
  incremental reindex. A periodic timer also fires (authoritative reconciliation
  for missed events and deletions).
- Three triggers — filesystem events (debounced), the periodic timer, and the
  on-demand `reindex` tool — funnel through one worker and the runtime lock, so
  mutations never overlap.
- `serve` starts the watcher automatically (config `reindex.watch`); the new
  `watch` command runs a headless indexing daemon.
- **Initialization is explicit.** `serve` never builds a first index for a repo;
  it only *refreshes* an index that already exists (`reindex.update_on_start`).
  A repo that was never indexed waits for an explicit `reindex`/`index` call —
  this avoids accidentally embedding a huge repo just by opening it. The watcher
  likewise ignores repos that are not yet initialized, so it starts contributing
  only after the first explicit index.
- Reads are lock-free: `search` grabs an immutable `NumpyVectorStore` snapshot;
  a reindex builds a fresh snapshot and assigns it atomically, so queries are
  never blocked by indexing.
- `index_status` reports `initialized` so a client can tell whether an explicit
  `reindex` is needed.

### Config

```toml
[reindex]
# refresh an already-initialized index at startup (never auto-creates one)
update_on_start = true
watch = true
interval_seconds = 300
debounce_seconds = 2.0
```

### Usage

```bash
gemma-embedder index  --root . --binary "$BIN"         # explicit first index
gemma-embedder watch  --root . --binary "$BIN"         # explicit daemon (indexes)
gemma-embedder watch  --root . --no-initial            # only update if inited
gemma-embedder serve  --root . --binary "$BIN"         # MCP + watcher (no auto-init)
```

### Verified

- Unit tests: `37 passed` (adds watcher debounce/periodic/ignore tests).
- Live MCP test: an edit becomes searchable within the debounce window; a
  deleted file disappears from the index; restarting with `update_on_start=false`
  preserves state (`last_reindex` unchanged, no full reindex) with the watcher
  still active.

---

## Using with opencode

Registered globally in `~/.config/opencode/opencode.json`:

```json
{
  "mcp": {
    "gemma-embedder": {
      "type": "local",
      "command": [
        "/home/joakim/projects/gemma-embedder/.venv/bin/gemma-embedder",
        "serve",
        "--root",
        "."
      ],
      "cwd": ".",
      "environment": {
        "GEMMA_EMBEDDER_BINARY": "/home/joakim/.local/share/gemma-embedder/llama.cpp-master/build-static/bin/llama-server"
      },
      "enabled": true,
      "timeout": 120000
    }
  }
}
```

- `cwd: "."` plus `--root .` means the server indexes whatever workspace
  opencode is opened in — the same global entry works for any repo.
- `GEMMA_EMBEDDER_BINARY` points at a **static** llama-server build (no local
  `.so` dependencies) under `~/.local/share/gemma-embedder/`, so it survives
  reboots and does not need `--binary` on the command line.
- `serve` completes the MCP handshake immediately. A repo that has already been
  indexed is refreshed in the background on startup (`update_on_start`); a repo
  that was never indexed is **not** touched until the client calls `reindex`.
  `index_status` reports `initialized` so an agent knows whether to initialize.
- Restart opencode after editing the config (config is not hot-reloaded).

The index is stored at `<repo>/.gemma-embedder/index.db`; add `.gemma-embedder/`
to the target repo's `.gitignore`. To disable the server for one project, add
`"mcp": { "gemma-embedder": { "enabled": false } }` to that repo's
`opencode.json`.
