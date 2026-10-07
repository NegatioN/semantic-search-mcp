# semantic-search-mcp

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
   gemma-embedder
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
- Output is Matryoshka-truncated to **256 dimensions** by default (near-lossless
  for code, ~3× smaller than the native 768).

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
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/.local/share/gemma-embedder/llama.cpp
cmake -S ~/.local/share/gemma-embedder/llama.cpp \
      -B ~/.local/share/gemma-embedder/llama.cpp/build \
      -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=ON
cmake --build ~/.local/share/gemma-embedder/llama.cpp/build --target llama-server -j "$(nproc)"
```

The resulting binary is
`~/.local/share/gemma-embedder/llama.cpp/build/bin/llama-server`. A static build
(`-DBUILD_SHARED_LIBS=OFF`) is recommended so the binary is relocatable and has
no local `.so` dependencies. Add `-DGGML_CUDA=ON` for a GPU build.

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
export GEMMA_EMBEDDER_BINARY="$HOME/.local/share/gemma-embedder/llama.cpp/build/bin/llama-server"
```

---

## First-time setup of a repository

**Nothing is indexed until you ask for it.** This is deliberate: opening a huge
repo should never trigger a surprise full embed.

Initialize a repo explicitly, either from the CLI:

```bash
.venv/bin/gemma-embedder index --root /path/to/repo
```

…or from inside your agent by asking it to initialize the index (the agent will
call the `reindex` tool, which does the same thing). You can confirm state with
`index_status`, which reports `initialized: true/false`.

After the first index, every subsequent visit **updates the existing index
incrementally** (`reindex.update_on_start`), and the watcher keeps it current as
you edit. Re-running `index` or calling `reindex` is always safe.

The index is stored at `<repo>/.gemma-embedder/index.db` — add `.gemma-embedder/`
to that repo's `.gitignore`.

---

## CLI usage

```bash
# environment check (binary + server reachability)
.venv/bin/gemma-embedder doctor

# build / refresh the index (explicit first index; incremental afterwards)
.venv/bin/gemma-embedder index  --root /path/to/repo
.venv/bin/gemma-embedder index  --root /path/to/repo --force      # re-embed all

# semantic search
.venv/bin/gemma-embedder search "where are auth tokens validated" -k 5 --root /path/to/repo
.venv/bin/gemma-embedder search "..." --json                      # machine-readable

# inspect the index
.venv/bin/gemma-embedder status --root /path/to/repo

# run a headless indexing daemon (initial index + watch)
.venv/bin/gemma-embedder watch --root /path/to/repo
.venv/bin/gemma-embedder watch --root /path/to/repo --no-initial  # only update if inited

# run the MCP server on stdio
.venv/bin/gemma-embedder serve --root /path/to/repo

# Phase-0 sanity probe against the model
.venv/bin/gemma-embedder probe
```

Add `--binary /path/to/llama-server` to any command, or rely on
`GEMMA_EMBEDDER_BINARY` / the `binary` config field. Use `--no-spawn` to target an
already-running server instead of spawning one.

---

## MCP usage (opencode)

Register the server in `~/.config/opencode/opencode.json`. `cwd: "."` plus
`--root .` makes the same entry index whatever workspace opencode is opened in:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "gemma-embedder": {
      "type": "local",
      "command": [
        "/abs/path/semantic-search-mcp/.venv/bin/gemma-embedder",
        "serve",
        "--root",
        "."
      ],
      "cwd": ".",
      "environment": {
        "GEMMA_EMBEDDER_BINARY": "/home/you/.local/share/gemma-embedder/llama.cpp/build/bin/llama-server"
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
`"mcp": { "gemma-embedder": { "enabled": false } }` to that repo's `opencode.json`.

---

## MCP tools

| Tool | Arguments | Returns |
| --- | --- | --- |
| `semantic_search` | `query`, `k=10`, `path?`, `min_score?`, `granularity?` (`any`/`file`/`symbol`) | Ranked hits: `path`, `start_line`, `end_line`, `score`, `snippet` |
| `get_context` | `path`, `start_line=1`, `end_line?`, `context_lines=20` | File lines, expanded by context, with `total_lines` |
| `reindex` | `path?`, `force=false` | Index report (`scanned`, `changed`, `unchanged`, `deleted`, `chunks`, `errors`) |
| `index_status` | — | `root`, `files`, `chunks`, `vectors`, `dim`, `initialized`, `model`, `last_reindex`, `reindex.watching` |

Typical agent flow: `semantic_search` → `get_context` on the best hit. On a
fresh repo: `index_status` → `reindex` → `semantic_search`.

---

## Configuration

Optional `gemma-embedder.toml` at the repo root (all fields have defaults):

```toml
[model]
repo = "ggml-org/embeddinggemma-2-GGUF:Q8_0"  # or :BF16
binary = ""                 # path to llama-server; empty = search PATH / env
manage_server = true        # spawn the server automatically
server_url = "http://127.0.0.1:8080"
ctx_size = 8192
gpu_layers = 0              # set 99 for GPU
normalize = true
dim = 256                   # MRL truncation: 256 | 512 | 768

[index]
roots = ["."]
respect_gitignore = true
max_file_bytes = 1000000
max_chunk_chars = 6000
overlap_chars = 600
snippet_chars = 1600

[store]
path = ".gemma-embedder/index.db"

[reindex]
update_on_start = true      # refresh existing index; never auto-creates one
watch = true
interval_seconds = 300
debounce_seconds = 2.0

[server]
transport = "stdio"
```

Environment overrides: `GEMMA_EMBEDDER_BINARY`, `GEMMA_EMBEDDER_SERVER_URL`.

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `unknown model architecture: 'gemma-embedding2'` | Your `llama-server` is too old. Build from upstream master (see above). |
| `no llama binary found` | Pass `--binary`, set `GEMMA_EMBEDDER_BINARY`, or put `llama-server` on `PATH`. |
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
src/gemma_embedder/
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
