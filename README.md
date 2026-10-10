# zemsearch

Local semantic code search powered by **EmbeddingGemma 2**, run natively with
**llama.cpp**, exposed to coding agents as an **MCP server**.

Point it at a repository and it embeds your code locally (no cloud, no Ollama),
keeps the index up to date as files change, and lets an agent query it by
meaning with `semantic_search` / `get_context`.

- **Fully local** — embeddings are computed by `llama-server` on your machine.
- **MCP-native** — `semantic_search`, `get_context`, `reindex`, `index_status`.
- **Explicit init** — a repo is never indexed automatically; you initialize it
  once, and it is incrementally refreshed afterwards.
- **Live index** — a filesystem watcher + periodic reconciliation keep it current.
- **Incremental** — only changed files are re-embedded (sha256 diffing).

The original project goal lives in [`GOAL.md`](GOAL.md); the long-form design
notes and phase history live in [`GUIDE.md`](GUIDE.md).

---

## How it works

```
llama-server (EmbeddingGemma 2 GGUF)  ──  /v1/embeddings, L2-normalized
        ▲
        │ HTTP
   zemsearch
   ├─ walker        gitignore-aware file discovery
   ├─ chunking      whole-file chunks (sliding window for large files)
   ├─ SQLite store  files / chunks / embeddings / meta (WAL)
   ├─ NumPy snapshot  exact dot-product search (vectors are unit-length)
   ├─ watcher       debounce + periodic reconciliation
   └─ MCP server    stdio transport
```

- Embeddings use the model's task prefixes: documents as
  `title: <path> | text: <code>` and queries as
  `task: code retrieval | query: <text>`.
- Vectors are L2-normalized, so **cosine, dot product, and L2 all rank
  identically**; the store uses the cheapest (dot).
- Vectors are **stored at the native 768 dimensions** (the model's actual output)
  and reduced to a **query dimension** (default 256) via the MRL *view*
  `normalize(v[:d])` at load/search time. Changing the query dimension never
  requires a reindex — only how many leading dims are read and compared.

---

## Requirements

- Linux or macOS, Python **3.12+**, and [`uv`](https://docs.astral.sh/uv/).
- A `llama-server` binary that supports the `gemma-embedding2` architecture.
  - The prebuilt `llama.app` installer currently ships an older build that fails
    with `unknown model architecture: 'gemma-embedding2'`. Build from upstream
    `llama.cpp` master until the prebuilt bucket catches up (see below).
- The EmbeddingGemma 2 GGUF weights (`ggml-org/embeddinggemma-2-GGUF`); the
  server downloads and caches them on first use.

### Building `llama-server`

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/.local/share/zemsearch/llama.cpp
cmake -S ~/.local/share/zemsearch/llama.cpp \
      -B ~/.local/share/zemsearch/llama.cpp/build \
      -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=ON
cmake --build ~/.local/share/zemsearch/llama.cpp/build --target llama-server -j "$(nproc)"
```

The resulting binary is
`~/.local/share/zemsearch/llama.cpp/build/bin/llama-server`. A static build
(`-DBUILD_SHARED_LIBS=OFF`) is recommended so the binary is relocatable and has
no local `.so` dependencies.

For a CUDA build (NVIDIA), configure a **separate output directory** and target
your GPU's compute capability so you only compile one architecture:

```bash
SRC=~/.local/share/zemsearch/llama.cpp
cmake -S "$SRC" -B "$SRC/build-cuda" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DLLAMA_CURL=ON
cmake --build "$SRC/build-cuda" --target llama-server -j "$(nproc)"
```

Then point the tool at it with `ZEMSEARCH_BINARY=$SRC/build-cuda/bin/llama-server`
(or `--binary`). On an RTX 4070 (sm_89) the CUDA server offloads the whole model
(~1.1 GB VRAM); for the 270M text backbone the gain is modest (query median
~19 ms → ~16 ms, best case ~17 ms → ~5 ms), so CPU is a fine default. `gpu_layers`
defaults to `0`, which lets `llama.cpp` auto-offload everything.

---

## Install

```bash
git clone git@github.com:NegatioN/semantic-search-mcp.git
cd semantic-search-mcp
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Make the binary location convenient (either use `--binary` per command, or set
it once):

```bash
export ZEMSEARCH_BINARY="$HOME/.local/share/zemsearch/llama.cpp/build/bin/llama-server"
```

---

## First-time setup of a repository

**Nothing is indexed until you ask for it.** This is deliberate: opening a huge
repo should never trigger a surprise full embed.

Initialize a repo explicitly, either from the CLI:

```bash
.venv/bin/zemsearch index --root /path/to/repo
```

…or from inside your agent by asking it to initialize the index (the agent will
call the `reindex` tool, which does the same thing). You can confirm state with
`index_status`, which reports `initialized: true/false`.

After the first index, every subsequent visit **updates the existing index
incrementally** (`reindex.update_on_start`), and the watcher keeps it current as
you edit. Re-running `index` or calling `reindex` is always safe.

The index is stored at `<repo>/.zemsearch/index.db` — add `.zemsearch/`
to that repo's `.gitignore`.

---

## CLI usage

```bash
# environment check (binary + server reachability)
.venv/bin/zemsearch doctor

# build / refresh the index (explicit first index; incremental afterwards)
.venv/bin/zemsearch index  --root /path/to/repo
.venv/bin/zemsearch index  --root /path/to/repo --force      # re-embed all

# semantic search
.venv/bin/zemsearch search "where are auth tokens validated" -k 5 --root /path/to/repo
.venv/bin/zemsearch search "..." --json                      # machine-readable

# inspect the index
.venv/bin/zemsearch status --root /path/to/repo

# run a headless indexing daemon (initial index + watch)
.venv/bin/zemsearch watch --root /path/to/repo
.venv/bin/zemsearch watch --root /path/to/repo --no-initial  # only update if inited

# run the MCP server on stdio
.venv/bin/zemsearch serve --root /path/to/repo

# Phase-0 sanity probe against the model
.venv/bin/zemsearch probe

# rough search-scaling microbenchmark (synthetic data, no model needed)
.venv/bin/zemsearch bench
.venv/bin/zemsearch bench --sizes 10000,100000,1000000 --json
```

Add `--binary /path/to/llama-server` to any command, or rely on
`ZEMSEARCH_BINARY` / the `binary` config field. Use `--no-spawn` to target an
already-running server instead of spawning one.

---

## MCP usage (opencode)

Register the server in `~/.config/opencode/opencode.json`. `cwd: "."` plus
`--root .` makes the same entry index whatever workspace opencode is opened in:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "zemsearch": {
      "type": "local",
      "command": [
        "/abs/path/semantic-search-mcp/.venv/bin/zemsearch",
        "serve",
        "--root",
        "."
      ],
      "cwd": ".",
      "environment": {
        "ZEMSEARCH_BINARY": "/home/you/.local/share/zemsearch/llama.cpp/build/bin/llama-server"
      },
      "enabled": true,
      "timeout": 120000
    }
  }
}
```

Restart opencode after editing the config (it is not hot-reloaded).

Then, in a new repo, either ask the agent to *"initialize the semantic index for
this repo"* before searching, or let it discover the state itself — the server's
instructions tell it to call `reindex` when `index_status` reports
`initialized: false`.

To disable the server for one project, add
`"mcp": { "zemsearch": { "enabled": false } }` to that repo's `opencode.json`.

### Companion skill

The repository ships an opencode skill at `skills/zemsearch/SKILL.md` that
teaches the agent *when* and *how* to use the tools (init gating, scoping, result
interpretation). Load it globally by pointing opencode at the `skills/` directory:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "skills": {
    "paths": ["/abs/path/semantic-search-mcp/skills"]
  }
}
```

opencode scans `skills.paths` recursively for `**/SKILL.md`, so the skill needs no
per-project copy. The MCP tool descriptions remain the portable source of truth;
the skill only adds opencode-specific workflow guidance.

---

## MCP tools

| Tool | Arguments | Returns |
| --- | --- | --- |
| `semantic_search` | `query`, `k=10`, `path?`, `min_score?`, `granularity?` (`any`/`file`/`symbol`), `diversity?` (MMR 0–1, 0 = plain top-k) | Ranked hits: `path`, `start_line`, `end_line`, `score`, `snippet` |
| `get_context` | `path`, `start_line=1`, `end_line?`, `context_lines=20` | File lines, expanded by context, with `total_lines` |
| `reindex` | `path?`, `force=false` | Index report (`scanned`, `changed`, `unchanged`, `deleted`, `chunks`, `errors`) |
| `index_status` | — | `root`, `files`, `chunks`, `vectors`, `native_dim`, `storage_dim`, `query_dim`, `loaded_dim`, `initialized`, `model`, `last_reindex`, `reindex.watching` |

Typical agent flow: `semantic_search` → `get_context` on the best hit. On a
fresh repo: `index_status` → `reindex` → `semantic_search`.

---

## Configuration

Optional `zemsearch.toml` at the repo root (all fields have defaults):

```toml
[model]
repo = "ggml-org/embeddinggemma-2-GGUF:BF16"  # native precision (or :Q8_0)
binary = ""                 # path to llama-server; empty = search PATH / env
manage_server = true        # spawn the server automatically
server_url = "http://127.0.0.1:8080"
ctx_size = 8192
gpu_layers = 0              # set 99 for GPU
normalize = true
query_dim = 256             # search dim (MRL view of native 768; no reindex needed)

[index]
roots = ["."]
respect_gitignore = true
max_file_bytes = 1000000
max_chunk_chars = 6000
overlap_chars = 600
snippet_chars = 1600
source = "file"             # "file" (whole-file walk) or "codegraph" (symbol graph)
codegraph_db = ".codegraph/codegraph.db"
codegraph_max_neighbors = 12
codegraph_max_body_chars = 2000

[store]
path = ".zemsearch/index.db"

[reindex]
update_on_start = true      # refresh existing index; never auto-creates one
watch = true
interval_seconds = 300
debounce_seconds = 2.0

[server]
transport = "stdio"
```

Environment overrides: `ZEMSEARCH_BINARY`, `ZEMSEARCH_SERVER_URL`.

### Code graph source (`source = "codegraph"`)

Instead of walking files, the index can mirror the symbol database built by
**[CodeGraph](https://github.com/colbymchenry/codegraph)** — a local, pre-indexed
code knowledge graph that parses a repo (20+ languages, native Rust kernel) and
auto-syncs an SQLite graph of every symbol, call edge and dependency as you edit
(`codegraph init` creates `.codegraph/codegraph.db`). CodeGraph is an *optional*
dependency: only the `codegraph` source needs it, and its database is read
read-only.

Each function/method/struct node becomes one `granularity="symbol"` hit whose
embedded text is **graph-enriched**: signature, docstring, 1-hop neighbours
(`calls`/`called by`/`creates`/`references`) and the source slice. That lets a
vector capture a symbol's *role*, so `semantic_search(..., granularity="symbol")`
returns exact symbols (`Client::CreateTable`) rather than whole files.

```toml
[index]
source = "codegraph"          # or "file" (default)
codegraph_db = ".codegraph/codegraph.db"
```

Sync is incremental and keyed on the graph, not the source tree:

- It **targets the database** — polling a fingerprint of `codegraph.db` and its
  WAL sidecars — and only reads when CodeGraph reports `index_state = complete`.
- It diffs each symbol's enriched-document hash against what is stored, then
  re-embeds only changed/new symbols and deletes vanished ones. Because the
  neighbourhood is part of the document, editing a *caller* also refreshes the
  callee even though the callee's file did not change.
- Switching `source`, or a codegraph extractor-version bump, triggers a full
  rebuild. `.codegraph/` is always excluded from file indexing.



## Performance

Search latency is dominated by two things:

- **Query embedding** — a single HTTP call to `llama-server`. Roughly constant
  (a few ms plus model warm-up) and independent of corpus size.
- **Vector search** — an exact dot product over every chunk, `O(n · dim)`,
  memory-bandwidth bound.

`zemsearch bench` measures the second part on synthetic data (no model or
index required). Numbers below are **CPU-only**, on an **AMD Ryzen 7 5700X
(8 cores / 16 threads)**, `dim=256`, `k=10`:

```
    chunks  dim   mem(MB)  build(ms)  search(ms)  filter+(ms)  search/s
     1,000  256       1.0       1.89       0.043        0.059     23201
    10,000  256      10.2      15.12       0.102        0.167      9783
   100,000  256     102.4     162.15       4.684        5.941       213
 1,000,000  256    1024.0    1830.20      54.785       58.819        18
```

Threading (measured by varying `OPENBLAS_NUM_THREADS` at 1M chunks):

- The dot product runs in NumPy/OpenBLAS and uses **~1–4 threads**, not all 16
  cores; it is **memory-bandwidth bound**, so extra cores stop helping quickly.
- The `path`/`granularity` filter is vectorized (`np.char.startswith` /
  array comparison), so it runs at C speed in the main thread.
- Query embedding (excluded from `bench`) runs in `llama.cpp`, which *is*
  multi-threaded. The runtime core count matters mostly there, not for search.

Other notes:

- **`path`/`granularity` filtering is now roughly on par with the dot product**
  (`filter+(ms)` ≈ `search(ms)`), and both scale linearly with `n`. It was
  previously a Python `O(n)` loop that cost ~3–4× more than the dot product at
  large `n` (e.g. ~177 ms → ~59 ms at 1M) — see
  `NumpyVectorStore.search` in `src/zemsearch/index/vectors.py`.
- **Memory** = `n · dim · 4` bytes for vectors (1 GB per million chunks at
  `dim=256`; 3 GB at `dim=768`), plus the precomputed metadata arrays: paths cost
  roughly `max_path_len · n` bytes and granularities ~`6 · n` bytes.
- **Dimension (`dim`)** affects storage and search cost, not embedding latency:
  `llama-server` always returns native 768-d vectors and the client truncates.
  See `reports/mrl-truncation-at-query.md` for the 256-vs-768 and
  query-time-truncation trade-offs.
- **`build`/reload** rebuilds the whole snapshot (vectors + metadata + `vstack`)
  and is paid on every reindex; constructing the metadata arrays adds some cost
  especially at 1M chunks.
- Unfiltered search is effectively free for real repos (sub-ms to a few ms);
  it only becomes noticeable past ~100k chunks.

So for typical repositories (thousands of chunks) exact brute-force search is
more than fast enough. Reach for ANN (e.g. `sqlite-vec`) only past a few hundred
thousand chunks.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `unknown model architecture: 'gemma-embedding2'` | Your `llama-server` is too old. Build from upstream master (see above). |
| `no llama binary found` | Pass `--binary`, set `ZEMSEARCH_BINARY`, or put `llama-server` on `PATH`. |
| Search returns nothing | The repo is not initialized. Call `reindex` (or `index`), then check `index_status`. |
| `input ... too large to process` | Raise `--batch-size`/`--ubatch-size`, or lower `max_chunk_chars`. |
| Server spawns but is unhealthy | Another process may own the port; the managed server picks a free port automatically. |
| Results feel stale after edits | Wait for the debounce (default 2s), or call `reindex` / lower `debounce_seconds`. |

---

## Development

```bash
.venv/bin/python -m pytest -q      # 40 tests
.venv/bin/ruff check src tests
```

Project layout:

```
src/zemsearch/
├── cli.py            # doctor | probe | index | watch | search | status | serve
├── config.py         # TOML config + env overrides
├── runtime.py        # shared wiring for CLI and MCP
├── languages.py      # extension -> language
├── model/            # prefixes, HTTP client, llama-server manager
├── chunking/         # chunk model + whole-file splitter
├── index/            # walker, SQLite store, NumPy vectors, indexer, watcher
└── mcp/server.py     # MCP tools (official SDK, stdio)
```

---

## Status

Phases 0–2 are implemented and verified:

- **P0** real end-to-end embedding probe (dim/norm/determinism + retrieval).
- **P1** whole-file index, SQLite store, NumPy exact search, MCP tools.
- **P2** live reindexing (watcher + periodic reconciliation) and explicit-init gating.

**P3 (next):** tree-sitter symbol-level chunking (`granularity="symbol"`, precise
line ranges) to improve precision over whole-file chunks.
**P4:** ANN/quantized vector store and 768-d rescoring. See [`GUIDE.md`](GUIDE.md).
