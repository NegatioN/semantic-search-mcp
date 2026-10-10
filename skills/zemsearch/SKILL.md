---
name: zemsearch
description: Use FIRST when exploring an unfamiliar codebase or looking for something you cannot name — i.e. before running ls/tree on a broad directory, or grepping a guessed keyword. Turns a vague question (where does auth live?, how does signup work?) into candidate files/symbols and, crucially, the codebase's own vocabulary, which you then feed into a normal grep/read. Phrase the query as a description of expected behavior, never a superlative ("most important", "fastest"). Also use to survey or onboard a repo. Use plain grep/read when you already know a concrete symbol, file, or exact string. Requires the zemsearch MCP tools.
---

# zemsearch — semantic code search

Local semantic search over an EmbeddingGemma 2 index, exposed as MCP tools. The
index lives at `<repo>/.zemsearch/index.db`; vectors are unit-normalized, so
scores are cosine similarity.

## When to use this skill

Reach for it **first** when:

- **You're about to explore by listing or guessing** — before `ls`/`tree` on a
  broad directory, or grepping a keyword you're not sure exists. One call finds
  the right area *and* the real names; browsing directories is slower and noisier.
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
- You just need a known, small directory's structure — a plain `ls` there is fine.

## Formulating the first query

The first query decides everything. Retrieval matches *meaning*, and in code the
meaning lives largely in **comments and docstrings**, so a query phrased as a
description of *what the code does* lands; a query phrased as *how important or
abstract the code is* does not.

**Rule: rewrite the user's question into the behavior you expect, in verbs.**

Abstract and superlative questions fail because their words are common across the
whole repo and match nothing in particular — "the most important part", "the
fastest path", "the core of the system". The same target is usually found
immediately when you instead describe the mechanism: what the code does, and
under what conditions.

Translate the initial question before searching:

| Initial question (do **not** query this literally) | Query the behavior instead |
| --- | --- |
| "What's the most performance-oriented part?" | "writes results straight to the consumer, skipping extra buffering when the data is already in memory" |
| "Where is the auth flow?" | "validates the user's credentials and issues a session token" |
| "How does request handling work?" | "matches an incoming request against patterns to pick a handler" |
| "What's the main config entry point?" | "loads settings from env vars and files, then applies defaults" |

Guidelines:

- **5–15 words of behavioral prose**, one mechanism per query. If the question
  spans several concerns, run several queries rather than one broad one.
- Use **concrete verbs the code would use** (`writes`, `parses`, `retries`,
  `caches`), not subsystem nouns (`server`, `router`, `pipeline`, `core`). Nouns
  that name a whole area match everything and drown the signal.
- **Never start from a superlative** ("fastest", "most critical", "best",
  "core"). If the user asked one, silently convert it into the behavior that
  would earn that superlative.
- The first query needs **no codebase vocabulary** — discovering it is the point.

## How to answer: candidate generation → lexical search

The highest-value pattern is **semantic search for candidate generation**:

1. **Find candidates and vocabulary.** Run `semantic_search` on the question
   rewritten as expected behavior (see *Formulating the first query*). It matches
   by *meaning*, so a query sharing none of the code's words still surfaces the
   relevant files — and, crucially, the **names the codebase actually uses**.
2. **Search again with the discovered terms.** Take the file paths and symbol
   names from the hits and use ordinary grep/read for exact, exhaustive results.
   Semantic search *finds* the vocabulary; lexical search *confirms and expands*.
3. **Read the code** at the hit (or its grep matches) with `get_context`.

Do not expect word-for-word matches — that is the point. Keep `k` generous on the
first pass, and use `path` to narrow follow-up searches.

## Worked example

Asked: *"explain the user authentication flow."*

1. Nothing is literally named "auth".
2. `semantic_search("user authentication flow", k=20)` →
   hits cluster on an internal component, e.g. `internal/identity/uzerflow.go`
   with symbol `UzerFlow`. **That is the codebase's term.**
3. `grep "UzerFlow"` for exact matches; `get_context` on the best ones.
4. Answer with `file:line` references to `UzerFlow`.

## Typical flow

```
index_status                                   # initialized?
reindex                                        # ONLY if not initialized — ask the user first
semantic_search("how does the user auth flow work", k=20)  # candidates + real term names
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
  Ranked hits: `path`, `start_line`, `end_line`, `score`, `snippet`. Use
  `granularity="symbol"` to get individual symbols rather than files (pair it with
  a larger `k` — see below). Set `diversity` (0–1) to spread results across more
  distinct files when generating candidates; `0` (default) is plain top-k.
- `get_context(path, start_line=1, end_line?, context_lines=20)`
  Verbatim lines for a range, plus `total_lines`.
- `reindex(path?, force=false)`
  Build/refresh the index. Reports `scanned`/`changed`/`unchanged`/`deleted`/`chunks`.
- `index_status()`
  `initialized`, `files`, `chunks`, native/storage/query dims, `last_reindex`.

### Choosing `k` (and granularity)

Be generous with `k`: the relevant code is rarely a single hit, and the extra
neighbours usually reveal the vocabulary you need. The right size depends on the
index's `granularity`:

- **Symbol chunks** (`granularity="symbol"`, when the index is built from a
  CodeGraph symbol graph): use **k = 15–25** (start at **20**). Symbol hits are
  fine-grained, so a wider window is what actually covers the topic.
- **Whole-file chunks** (default): **k ≈ 8–12** is plenty; files are coarse, so
  more hits mostly add noise.

If the extra hits are near-duplicates, raise `diversity` (0.3–0.5) rather than
`k`. If unrelated files creep in, prefer **rephrasing the query behaviorally**
(see above) or narrowing with `path` — a raw `min_score` floor is a weak fix,
because cosine scores cluster in a narrow band and a floor easily discards good
hits along with the noise.

### Choosing `diversity`

`diversity` is MMR strength: `0` = plain top-k (default); higher trades a little
relevance for a wider spread. It does **not** scale with repo size — it depends
on how redundant the candidates are and on `k`. The aim is to cover the query's
relevant topics roughly in proportion to their relevance (MMR is a greedy
approximation of that xQuAD/α-nDCG ideal).

Calibration: the common library default is *balanced* MMR (LangChain
`lambda_mult=0.5`), which is ranking-equivalent to about `diversity ≈ 1.0` on our
scale; a relevance-leaning `λ=0.7` is about `diversity ≈ 0.4`. In practice start
around **0.3–0.5**: raise it when the top-k is near-duplicates, lower it when
unrelated files appear. `diversity` reshapes a result set — it does **not** make
a badly-phrased query relevant; fix the query text first. On small or
heterogeneous repos keep it low.

## More examples

- *"How does request handling flow through this service?"* → `semantic_search`
  → note the naming convention it reveals (e.g. `Middleware`, `Pipeline`) →
  grep those → `get_context`.
- *"Where does the benchmark report get generated?"* (fresh repo) →
  `index_status` shows `initialized: false` → ask the user → `reindex` →
  `semantic_search` → `get_context`.
- *"Find the symbol `parseConfig`."* → grep, not this skill.

## Pitfalls

- **Superlative or abstract query** → "the most important part", "the fastest
  path", "the core of X" return junk (docs, examples, symbols that merely share a
  common word). Rewrite as a behavioral description before concluding the code
  isn't there. This is the most common failure mode.
- **Subsystem nouns** → words like `server`, `router`, `pipeline`, `config`,
  `path` are everywhere and dominate the ranking. Describe the *action* instead.
- **Examples, tests, and benchmarks rank high** → for "where is X implemented",
  sample/benchmark/test files often outrank the real source. Narrow with `path`
  (e.g. the main source tree) or mentally filter `test`/`example`/`benchmark`
  paths before trusting a hit.
- **Empty results on a valid query** → check `granularity`. Only some values are
  populated for a given index; an unsupported one can silently return **zero**
  hits rather than an error. Try `granularity="any"` (or omit it) when this
  happens.
- **Empty results on a fresh repo** → not initialized; ask the user, then `reindex`.
- **Stale right after edits** → the watcher reindexes after a short debounce
  (~2s); wait a moment or call `reindex`.
- **Huge repo** → the first index can take minutes; scope with `path` or exclude
  directories via a `zemsearch.toml` at the repo root.
- **Chunk kinds** → with the default `file` source, `start_line`/`end_line` may
  span a whole file — open hits with `get_context`. With a `codegraph` source,
  hits are individual symbols. Either way, prefer grepping the discovered terms
  once you have them.
- **Index dir** → `<repo>/.zemsearch/`; add it to the repo's `.gitignore`.

## Tuning

Optional `zemsearch.toml` at the repo root controls excludes, chunk size,
watch interval, and more. See the project README.
