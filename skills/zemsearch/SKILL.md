---
name: zemsearch
description: Use FIRST when a request is broad or vague and how it maps onto this codebase is unclear — for example a flow, a feature, or an area with no obvious name. Semantic search surfaces candidate files and, importantly, the codebase's own vocabulary (an internal name like `UzerFlow`), which you then feed into a normal grep/read. Also use to survey an unfamiliar repo. Use plain grep/read when you already know a concrete symbol, file, or exact string. Requires the zemsearch MCP tools.
---

# zemsearch — semantic code search

Local semantic search over an EmbeddingGemma 2 index, exposed as MCP tools. The
index lives at `<repo>/.zemsearch/index.db`; vectors are unit-normalized, so
scores are cosine similarity.

## When to use this skill

Reach for it **first** when:

- The request is **broad or vague** and its interpretation in this codebase is
  unclear — "how does the user auth flow work?", "where does billing live?",
  "what happens on signup?" — even when none of your words appear in the code.
- You need **candidate files/symbols**, or the codebase's **internal vocabulary**,
  to drive a normal search (this is the main pattern — see below).
- You're **surveying or onboarding** an unfamiliar area.
- You need to check or initialize the index.

Use plain grep/read directly when:

- You already know a concrete symbol, function, file, or exact string.
- It's a mechanical, well-scoped edit with a known location.

## How to answer: candidate generation → lexical search

The highest-value pattern is **semantic search for candidate generation**:

1. **Find candidates and vocabulary.** Run `semantic_search` on the raw question.
   It matches by *meaning*, so a query sharing none of the code's words still
   surfaces the relevant files — and, crucially, the **names the codebase
   actually uses**.
2. **Search again with the discovered terms.** Take the file paths and symbol
   names from the hits and use ordinary grep/read for exact, exhaustive results.
   Semantic search *finds* the vocabulary; lexical search *confirms and expands*.
3. **Read the code** at the hit (or its grep matches) with `get_context`.

Do not expect word-for-word matches — that is the point. Scope with `path` and
`k` to keep candidates focused.

## Worked example

Asked: *"explain the user authentication flow."*

1. Nothing is literally named "auth".
2. `semantic_search("user authentication flow", k=8)` →
   hits cluster on an internal component, e.g. `internal/identity/uzerflow.go`
   with symbol `UzerFlow`. **That is the codebase's term.**
3. `grep "UzerFlow"` for exact matches; `get_context` on the best ones.
4. Answer with `file:line` references to `UzerFlow`.

## Typical flow

```
index_status                                   # initialized?
reindex                                        # ONLY if not initialized — ask the user first
semantic_search("how does the user auth flow work", k=8)   # candidates + real term names
grep "<discovered term>"                        # precise search with the codebase's vocabulary
get_context(path=<best hit>, start_line=..., end_line=...)
```

## Initialization

If `index_status` reports `initialized: false`, the repo has **never** been
indexed. **Ask the user before running `reindex`** — a first index can be
expensive on large repos. You can scope it, e.g. `reindex(path="src/")`. Once
initialized, updates are automatic (file watcher + periodic reconciliation), so
you never need to re-init.

## Tools

- `semantic_search(query, k=10, path?, min_score?, granularity?, diversity?)`
  Ranked hits: `path`, `start_line`, `end_line`, `score`, `snippet`. Set
  `diversity` (0–1) to spread results across more distinct files when generating
  candidates; `0` (default) is plain top-k.
- `get_context(path, start_line=1, end_line?, context_lines=20)`
  Verbatim lines for a range, plus `total_lines`.
- `reindex(path?, force=false)`
  Build/refresh the index. Reports `scanned`/`changed`/`unchanged`/`deleted`/`chunks`.
- `index_status()`
  `initialized`, `files`, `chunks`, native/storage/query dims, `last_reindex`.

### Choosing `diversity`

`diversity` is MMR strength: `0` = plain top-k (default); higher trades a little
relevance for a wider spread. It does **not** scale with repo size — it depends
on how redundant the candidates are and on `k`. The aim is to cover the query's
relevant topics roughly in proportion to their relevance (MMR is a greedy
approximation of that xQuAD/α-nDCG ideal).

Calibration: the common library default is *balanced* MMR (LangChain
`lambda_mult=0.5`), which is ranking-equivalent to about `diversity ≈ 1.0` on our
scale; a relevance-leaning `λ=0.7` is about `diversity ≈ 0.4`. In practice start
around **0.3–0.5**: raise it when the top-k is near-duplicates, lower it (or add
a `min_score` floor) when unrelated files appear. On small or heterogeneous repos
keep it low.

## More examples

- *"How does request handling flow through this service?"* → `semantic_search`
  → note the naming convention it reveals (e.g. `Middleware`, `Pipeline`) →
  grep those → `get_context`.
- *"Where does the benchmark report get generated?"* (fresh repo) →
  `index_status` shows `initialized: false` → ask the user → `reindex` →
  `semantic_search` → `get_context`.
- *"Find the symbol `parseConfig`."* → grep, not this skill.

## Pitfalls

- **Empty results on a fresh repo** → not initialized; ask the user, then `reindex`.
- **Stale right after edits** → the watcher reindexes after a short debounce
  (~2s); wait a moment or call `reindex`.
- **Huge repo** → the first index can take minutes; scope with `path` or exclude
  directories via a `zemsearch.toml` at the repo root.
- **Whole-file chunks (current)** → `start_line`/`end_line` may span a whole file;
  open hits with `get_context`, and prefer grepping the discovered terms once you
  have them.
- **Index dir** → `<repo>/.zemsearch/`; add it to the repo's `.gitignore`.

## Tuning

Optional `zemsearch.toml` at the repo root controls excludes, chunk size,
watch interval, and more. See the project README.
