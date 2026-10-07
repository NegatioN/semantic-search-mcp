---
name: gemma-embedder
description: Use when a question about the code is vague or conceptual — how or where some behavior, feature, or concept lives in the codebase — rather than asking about a concrete symbol, file, or exact string. Examples include "how are retries handled?", "where does the billing logic live?", "what generates the report?". Reach for this skill FIRST to get an overview of candidate code, then get_context to read the best hits; fall back to grep/read for exact identifiers. Also use it to check or initialize the semantic index (index_status / reindex) before searching. Requires the gemma-embedder MCP tools.
---

# gemma-embedder — semantic code search

Local semantic code search backed by EmbeddingGemma 2, exposed as MCP tools.
The index lives at `<repo>/.gemma-embedder/index.db`. Vectors are
L2-normalized, so scores are cosine similarity and dot-product ranking.

## When to use this skill

Reach for this skill **first** when:

- The question is about a **concept, behavior, or feature**, and you don't yet
  know the symbol or file. E.g. "how does retry/backoff work?", "where is
  authentication enforced?", "what produces the benchmark report?".
- You want a **map of candidate locations** before reading code, or the repo is
  unfamiliar.
- You need to **initialize or refresh** a repo (first visit, or after large
  changes) before searching.

Do **not** use this skill when:

- The question names a **concrete symbol, function, file, or exact string** — use
  grep/read directly (exact and fast).
- You already know which file(s) to read.
- It is a mechanical, well-scoped edit with a known location.

Order of operations: **locate** with semantic search → **read** with
`get_context` → then other tools. If a concrete symbol emerges mid-investigation,
switch to grep/read for precision.

## Golden rules

1. **Never assume the repo is indexed.** Call `index_status` first. Search on an
   uninitialized repo returns empty — that does not mean the code is absent.
2. **A fresh repo must be initialized explicitly.** If `index_status` reports
   `initialized: false`, call `reindex` before `semantic_search`. This is
   intentional: opening a large repo must not trigger a surprise full embed.
3. **Search by meaning, then read.** Use `semantic_search`, then `get_context`
   on the best hit(s) to read the surrounding lines.
4. **Semantic search is not grep.** For an exact identifier, symbol name, string,
   or regex, use ordinary grep/read. Use these tools for concepts and intent.
5. **Scope and size sensibly.** Pass `path` to restrict to a subtree and `k` to
   limit results (5–10 is usually enough). Lower precision in, better results out.
6. **Don't over-trust weak scores.** Scores live in roughly `[0, 1]`; below ~0.6
   is usually a poor match. Prefer the top hit only when it clearly dominates.
7. **Edits are picked up automatically.** A file watcher reindexes changed files
   (a couple of seconds, debounced). If results look stale, call `reindex`.

## First time in a repo

```
index_status                       # -> initialized: false
reindex                            # build the index (may take a while on big repos)
index_status                       # -> initialized: true, files/chunks counts
semantic_search(query="...", k=5)  # now works
```

`reindex` accepts `path` to initialize only a subtree, e.g.
`reindex(path="src/")`, useful for very large repos.

## Every subsequent visit

The index is refreshed automatically on server start and kept current by the
watcher, so you normally just:

```
index_status          # confirm initialized: true
semantic_search(...)
get_context(...)
```

Call `reindex` when you want to force an update, initialize a subtree, or use
`force=True` to re-embed everything after a large change.

## Tools

- `semantic_search(query, k=10, path=?, min_score=?, granularity="any")`
  Returns ranked hits: `path`, `start_line`, `end_line`, `score`, `snippet`.
  `granularity` is `any` | `file` | `symbol` (symbol mode is not implemented yet).
- `get_context(path, start_line=1, end_line=?, context_lines=20)`
  Reads a line range, expanded by `context_lines`. Use it to read a hit in full.
- `reindex(path=?, force=false)`
  Build/refresh the index. Returns `scanned`, `changed`, `unchanged`, `deleted`,
  `chunks`, `errors`.
- `index_status()`
  `root`, `files`, `chunks`, `vectors`, `dim`, `initialized`, `model`,
  `last_reindex`, `reindex.watching`.

## Examples

User: "Where does this project validate auth tokens?"
- `semantic_search("where auth tokens are validated and parsed", k=5)`
- `get_context(path=<best hit>, start_line=<start>, end_line=<end>)`
- Answer with `file:line` references.

User: "How is the benchmark workload generated?" (fresh repo, empty results)
- `index_status` → `initialized: false`
- `reindex` (tell the user the first index may take a moment)
- `semantic_search("generate the benchmark workload of simulated users", k=5)`
- `get_context(...)` on the top hit.

User: "Find the exact symbol `parseConfig`."
- Prefer grep, not semantic search. Only reach for `semantic_search` if grep is
  unhelpful or the user means "the config-parsing logic" rather than the literal name.

## Pitfalls

- **Empty results on a fresh repo** → not initialized; run `reindex`.
- **Empty results right after `reindex`** → indexing may still be running in the
  background; re-check `index_status` (`files`/`vectors` growing).
- **Huge repo** → `reindex` can take minutes and load the CPU/GPU. Scope with
  `path`, or exclude directories via a `gemma-embedder.toml` in the repo root.
- **Whole-file chunks (current)** → `start_line`/`end_line` often span the whole
  file until symbol-level chunking lands; a snippet may be large.
- **Store is untracked** → the index is written to `<repo>/.gemma-embedder/`; add
  that directory to the repo's `.gitignore`.

## Tuning (optional)

Drop a `gemma-embedder.toml` in the repo root to override defaults, e.g. reduce
`dim`, change `context` exclusions, chunk sizes, or `reindex.debounce_seconds`.
See the project README for the full reference.
