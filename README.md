# zemsearch

Local semantic code search, exposed to coding agents as an **MCP server**.

Point it at a repository and it embeds your code locally keeps the index up to date as files change, and lets an agent query it by
meaning with `semantic_search` / `get_context`.

- **Fully local** — embeddings are computed by `llama-server` on your machine.
- **MCP-native** — `semantic_search`, `get_context`, `reindex`, `index_status`.
- **Explicit init** — a repo is never indexed automatically; you initialize it once, and it is incrementally refreshed afterwards.
- **Live index** — a filesystem watcher + periodic reconciliation keep it current.
- **Incremental** — only changed files are re-embedded (sha256 diffing).

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

Embeddings come from **EmbeddingGemma 2** served by `llama-server`. The model
needs its task prefixes and has a few sharp edges (precision, MRL truncation,
its 8K context window) — see **[docs/model.md](docs/model.md)**.

---

## Requirements

- Linux or macOS, Python **3.12+**, and [`uv`](https://docs.astral.sh/uv/).
- A `llama-server` binary that supports the `gemma-embedding2` architecture.
  Install the default build (see below); if the build you get is too old it exits
  with `unknown model architecture: 'gemma-embedding2'`, in which case build from
  upstream `llama.cpp` master.
- The EmbeddingGemma 2 GGUF weights (`ggml-org/embeddinggemma-2-GGUF`); the
  server downloads and caches them on first use.

### Installing `llama-server`

EmbeddingGemma 2 requires a recent `llama.cpp` (the `LLM_ARCH_GEMMA_EMBEDDING2`
architecture, on `main`). Pre-built installs, from the
[official install docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md):

| Platform | Command |
| --- | --- |
| macOS / Linux (Homebrew) | `brew install llama.cpp` |
| Windows (winget) | `winget install llama.cpp` |
| conda / pixi / mamba | `conda install -c conda-forge llama.cpp` |
| Nix (macOS / Linux) | `nix profile install nixpkgs#llama-cpp` |
| Linux, no package manager | `curl -LsSf https://llama.app/install.sh \| sh` |
| Docker (nothing to install) | `docker run ghcr.io/ggml-org/llama.cpp:server ...` |

`zemsearch` auto-detects either the unified `llama` binary (Homebrew, the
`llama.app` installer — invoked as `llama serve`) or the standalone
`llama-server` on your `PATH`, so no `--binary` is needed. Override with
`--binary` or `ZEMSEARCH_BINARY` to use a specific build.

**Version caveat:** package managers track tagged *releases*, and this
architecture currently lives on `main`, so a packaged build may still fail with
`unknown model architecture: 'gemma-embedding2'` until a release includes it
(later, `brew upgrade llama.cpp`, or `llama update` for the app CLI). If you hit
that now, build from source:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/.local/share/zemsearch/llama.cpp
cmake -S ~/.local/share/zemsearch/llama.cpp \
      -B ~/.local/share/zemsearch/llama.cpp/build \
      -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=ON
cmake --build ~/.local/share/zemsearch/llama.cpp/build --target llama-server -j "$(nproc)"
```

The resulting binary is
`~/.local/share/zemsearch/llama.cpp/build/bin/llama-server`. A static build
(`-DBUILD_SHARED_LIBS=OFF`) is relocatable and has no local `.so` dependencies.

---

## Install

```bash
git clone git@github.com:NegatioN/zemsearch.git
cd zemsearch
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

If `llama-server` isn't already on your `PATH` (e.g. you built from source),
point the tool at it once:

```bash
export ZEMSEARCH_BINARY="$HOME/.local/share/zemsearch/llama.cpp/build/bin/llama-server"
```

---

## First-time setup

Index a repository once:

```bash
.venv/bin/zemsearch index --root /path/to/repo
```

That's all — from then on the index keeps itself up to date as you edit. (Agents
can do the same by calling the `reindex` tool.) The index lives in
`<repo>/.zemsearch/`; add that to the repo's `.gitignore`.

---

## CLI usage

```bash
.venv/bin/zemsearch index  --root /path/to/repo   # build/refresh the index (incremental)
.venv/bin/zemsearch search "where are auth tokens validated" --root /path/to/repo
.venv/bin/zemsearch status --root /path/to/repo   # inspect the index
.venv/bin/zemsearch serve  --root /path/to/repo   # MCP server (stdio)
```

Add `--binary /path/to/llama-server` (or set `ZEMSEARCH_BINARY`); `--no-spawn`
targets an already-running server. `zemsearch --help` lists the rest
(`doctor`, `watch`, `bench`, `probe`).

---

## Using with a coding agent

zemsearch ships two pieces to wire into your agent:

- an **MCP server** — add a local stdio server that runs `zemsearch serve --root .`,
  giving the agent `semantic_search`, `get_context`, `reindex`, and `index_status`;
- a **companion skill** — point the agent at `skills/zemsearch/` so it knows when
  and how to use those tools.

The most common setup is **opencode**; its full config, along with notes for other
agents, is in **[docs/agents.md](docs/agents.md)**.

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

### What gets indexed (ignore model)

For `source = "file"`, discovery uses one gitignore-semantics matcher shared by
the walker and the file watcher. It is seeded, in order, with the built-in
`exclude` defaults (dependencies/build output/caches across ecosystems), then
every `.gitignore` under the root, then git's own root-relative exclude files
(`.git/info/exclude` and `core.excludesFile`). Negations (`!pattern`) are honored,
and a bare `dir/` pattern matches at any depth. Re-include anything the defaults
skip with a `!` rule in your `.gitignore` or in `exclude`. When
`respect_gitignore = false`, only `exclude` applies.

### Optional: embedding a CodeGraph symbol graph

If you already index your repository with
**[CodeGraph](https://github.com/colbymchenry/codegraph)** — a local, pre-indexed
code knowledge graph — zemsearch can embed its parsed symbols instead of whole
files, giving exact symbol-level hits (`Client::CreateTable`) rather than
file-level matches.

Set `index.source = "codegraph"`. See **[docs/codegraph.md](docs/codegraph.md)**
for configuration and how the graph-triggered incremental sync works.

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
.venv/bin/python -m pytest -q 
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
