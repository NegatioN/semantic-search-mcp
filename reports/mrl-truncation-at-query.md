# Report: persisting 768-d vectors and truncating at query time

**Question.** Instead of persisting vectors already truncated to the working
dimension (default 256), can we persist the full native 768-d vectors and slice
(`[:, :d]`) at query time? This would let us change the retrieval dimension
without re-embedding. What does it cost?

**Method.** `scripts/mrl_truncation_probe.py` compares three strategies for a
`dim=d` query over unit-normalized vectors:

| | stored | per-query work |
| --- | --- | --- |
| **A** baseline | `(n, d)` truncated + row-normalized | contiguous `A @ q` |
| **B** query-time | `(n, 768)` | `(V[:, :d] @ q) / row_norm`, row norms cached per `d` |
| **C** query-time (naive) | `(n, 768)` | same as B but recompute row norms every query |
| **D** lazy cache | `(n, 768)` | first query at `d` materializes + caches the normalized `(n, d)` matrix, then A-speed matvec |
| **E** prefix load | `(n, 768)` on disk | when `d` is known, read only the leading `d` floats per vector (`substr`) into an `(n, d)` matrix — the 768 copy never enters RAM |

Vectors come from the real `clickhouse-tests` index at 768 (64 chunks); scaling
numbers use synthetic `(n, 768)` row-normalized matrices. Pure search time
excludes one-time build. Machine: AMD Ryzen 7 5700X, CPU-only. Treat the numbers
as indicative — this host shows ±20% run-to-run variance at large `n`.

## Correctness

Query-time truncation is **rank-identical** to persisting at `d`, for every
tested dimension (on the real vectors):

```
correctness dim=128: baseline vs query-time top-10 identical = True
correctness dim=256: baseline vs query-time top-10 identical = True
correctness dim=512: baseline vs query-time top-10 identical = True
correctness dim=768: baseline vs query-time top-10 identical = True
```

This is expected: MRL truncation then re-normalization is exactly what the ingest
path already does, so slicing later produces the same vectors (up to float
rounding). **D** builds the same normalized matrix as **A**, so it is identical
by construction.

## Search time (ms) and memory

`dim=256` (the working default). `D cold` is the first query at `d` (includes the
one-time materialize+normalize); `D warm` is a cached query:

| n | A | B | C | D cold | D warm | mem A | mem D |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.78 | 3.34 | 7.80 | 13.4 | 1.09 | 10 MB | 41 MB |
| 100,000 | 40.8 | 61.4 | 99.4 | 103.7 | 41.2 | 102 MB | 410 MB |
| 300,000 | 111.0 | 165.9 | 261.8 | 281.3 | 103.6 | 307 MB | 1229 MB |

`mem D` = 768 store + cached `(n, d)` matrix (and grows with each distinct `d`
you cache).

Full-dimension reference (`dim=768`), where A and B converge and D's cache is a
copy:

| n | A | B | C | D cold | D warm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 120.2 | 115.9 | 181.6 | 215.3 | 107.7 |
| 300,000 | 293.0 | 308.8 | 473.0 | 616.5 | 288.6 |

## Loading only the needed dims (E)

`scripts/mrl_lazy_load_probe.py` stores 768-d blobs in SQLite and compares a full
load against `SELECT substr(vector, 1, d*4)`. Peak allocation and bytes read
scale with `d`, not 768:

| n | load | peak | read | time |
| ---: | --- | ---: | ---: | ---: |
| 100,000 | full 768 | 650 MB | 307 MB | 1020 ms |
| 100,000 | prefix d=128 | 138 MB | 51 MB | 778 ms |
| 100,000 | prefix d=256 | 241 MB | 102 MB | 773 ms |
| 100,000 | prefix d=512 | 446 MB | 205 MB | 813 ms |

(20k for reference: full 130 MB / 61 MB / 209 ms; d=256 48 MB / 20 MB / 163 ms.)

- **Memory tracks `d`**, not 768: at 100k a `d=256` prefix load peaks at 241 MB
  vs 650 MB for the full load (~2.7× less); steady-state after the per-row
  buffers are released is just the `(n, d)` matrix (~102 MB).
- **Disk reads track `d`** (307 → 102 MB), because `substr` returns only the
  leading bytes of each blob.
- **Time does not drop proportionally** (1020 → 773 ms): the per-row Python loop
  (`np.frombuffer` + `vstack`) and SQLite b-tree traversal dominate, not raw
  bytes. A single contiguous blob or a memory-mapped flat file would make E
  near-I/O-bound.
- Correctness is exact: `prefix == full[:, :d]` asserted for every dim.

**Trade-off vs D:** E never holds the 768 array in RAM, so it uses `d`-sized
memory when `d` is known; the cost is re-reading from disk (~0.8 s per 100k) to
switch `d`. D pays 3× RAM to make dimension changes instant.

## Analysis

- **B is ~1.5× slower than A at `dim=256`.** `V[:, :d]` is a *strided* view
  (row stride = 768 floats), so the matvec touches the same cache lines with
  worse locality than a contiguous `(n, d)` matrix, and there is an extra
  per-row divide. At `d=768` the view is contiguous and B ≈ A.
- **C is ~2.7× slower than A** because it adds a full `O(n·d)` norm pass on top
  of the strided dot product, every query.
- **D warm ≈ A** (the cached matrix is exactly A's matrix): once materialized,
  queries cost the same as persisting at `d`. The price is paid up front on the
  first query per `d` (**D cold − D warm** ≈ one `O(n·d)` slice+normalize: ~60 ms
  at 100k, ~180 ms at 300k for `d=256`).
- **Storage is the trade-off:** D keeps the 768 store *and* each cached
  `(n, d)` matrix (e.g. 410 MB at 100k for `d=256`, vs 102 MB for A). Multimodal
  multi-dim use multiplies the cached matrices.
- **Embedding latency is unchanged either way**: `llama-server` always returns
  768-d vectors, so truncation already happens client-side. `dim` only affects
  stored size and search.

## Recommendation

- **If you only ever query one dimension, keep persisting at that dimension
  (256).** It is the fastest to search and 3× smaller. Changing `dim` is a
  one-time re-embed (cheap for small repos; parallelizable).
- **If you need to switch dimensions without re-embedding, use D**: persist 768
  once and lazily materialize + cache a row-normalized `(n, d)` matrix per
  requested `d`. Warm searches then match A; you pay one `O(n·d)` build per dim
  plus the extra cached matrices.
- **If RAM is the constraint and dimension changes are rare, use E**: persist 768
  but load only the leading `d` dims at startup (`substr`). Memory ≈ A and disk
  reads ≈ A's, while the 768 vectors stay on disk. The cost is a disk re-read
  (~0.8 s per 100k) when you change `d`.
- **Faster E**: store vectors as one contiguous blob (or memory-mapped flat file)
  instead of per-row blobs, so the leading-dims load is a single contiguous read
  with no per-row Python loop.
- **Do not** compute truncation inside the hot search path (B/C): the strided
  view ~1.5× penalty and C's per-query norm pass are pure waste when a cached
  `(n, d)` matrix gives A speed.

## Reproduce

```bash
.venv/bin/gemma-embedder index --root ~/projects/clickhouse-tests \
    --config /tmp/opencode/dim768.toml      # store full 768-d vectors
.venv/bin/python scripts/mrl_truncation_probe.py   # strategies A/B/C/D (in-RAM)
.venv/bin/python scripts/mrl_lazy_load_probe.py    # strategy E (prefix load from SQLite)
```

(`/tmp/opencode/dim768.toml` sets `[model] dim = 768` and a separate
`[store] path` so the existing 256 index is left untouched.)
