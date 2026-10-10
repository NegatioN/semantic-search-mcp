# Embedding a CodeGraph symbol graph

By default zemsearch indexes whole files. If a repository is already parsed by
**[CodeGraph](https://github.com/colbymchenry/codegraph)** — a local, pre-indexed
code knowledge graph — zemsearch can embed its symbols directly instead. This
yields exact symbol-level results (`Client::CreateTable`) inside `semantic_search`
rather than whole-file hits.

CodeGraph is an *optional* dependency. Only the `codegraph` source needs it, and
its database is read read-only.

## What gets embedded

One embedding per node **except** the structural kinds `file` and `import` — a
*denylist*, so language-specific declarations (`class`, `trait`, `module`,
`object`, `enum`, …) are picked up automatically instead of being dropped by a
Go/TS-shaped allowlist. Nodes are deduplicated by
`(file_path, qualified_name, kind)`. Each is stored as a `granularity="symbol"`
chunk whose document is **graph-enriched**: the symbol's signature, docstring, its
1-hop neighbourhood (`calls`, `called by`, `creates`, `references`) and the source
slice. Folding the neighbourhood into the embedded text is what lets a vector
capture a symbol's *role* and not just its body.

A real example document:

```
title: internal/ch/client.go::Client::CreateTable | text: method Client::CreateTable
signature: CreateTable(ctx context.Context, v variant.Variant) error
doc: CreateTable creates the variant's table and any aggregate view.
returns: error
file: internal/ch/client.go
calls: CreateStatements, DropStatements
called by: run
references: Variant
func (c *Client) CreateTable(ctx context.Context, v variant.Variant) error {
	for _, stmt := range v.DropStatements() {
		if err := c.conn.Exec(ctx, stmt); err != nil {
			return fmt.Errorf("ch: drop: %w", err)
		}
	}
	...
}
```

See the
[EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2)
for the `title: … | text: …` / `task: code retrieval` prompt format.

## Configuration

```toml
[index]
source = "codegraph"                    # "file" (default) or "codegraph"
codegraph_db = ".codegraph/codegraph.db"
codegraph_max_neighbors = 12            # 1-hop neighbours folded into each document
codegraph_max_body_chars = 2000         # cap on the appended source slice
```

## How sync works

Sync is incremental and keyed on the graph, not the source tree:

- It **targets the database** — polling a fingerprint of `codegraph.db` and its
  WAL sidecars — and only reads once CodeGraph reports `index_state = complete`,
  so a half-written graph is never ingested.
- It diffs each symbol's **enriched-document hash** against what is stored, keyed
  by `(path, symbol, kind)`, then re-embeds only changed or new symbols and
  deletes the ones that vanished.
- Because the neighbourhood is part of the document, editing a *caller* also
  refreshes the callee even though the callee's own file did not change.
- Switching `source`, or a CodeGraph extractor-version bump, triggers a full
  rebuild. `.codegraph/` is always excluded from file indexing.

## Requirements

1. Install CodeGraph and initialize the target repo, which creates
   `.codegraph/codegraph.db`:

   ```bash
   codegraph init
   ```

   CodeGraph keeps the graph fresh automatically; zemsearch then mirrors it.
2. Set `index.source = "codegraph"` (see above).

## Estimating index size

`scripts/codegraph_embed_count.py` reports how many embeddings a given CodeGraph
database would produce and the approximate vector-scan latency for that count,
without running the model:

```bash
python scripts/codegraph_embed_count.py /path/to/repo
```
