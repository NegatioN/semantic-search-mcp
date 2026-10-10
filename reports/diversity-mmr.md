# Report: diversity knob (MMR) — effect and speed

**What it is.** A single optional knob on `semantic_search` that trades a little
relevance for a wider variety of candidates, using Maximal Marginal Relevance:
each pick must be relevant to the query *and* not a near-duplicate of what's
already picked. `diversity=0` is exactly the old top-k behavior, so the feature
is opt-in and backward compatible.

Interface: `semantic_search(query, …, diversity=0.0)`, CLI
`zemsearch search … --diversity 0.4`, and the `bench` `div(ms)` column.

## Speed (synthetic, `dim=256`, `k=10`, `diversity=0.3`)

| chunks | search | + diverse | overhead |
| ---: | ---: | ---: | ---: |
| 10,000 | 0.251 ms | 0.728 ms | +0.48 ms (~2.9×, but sub-ms) |
| 100,000 | 5.08 ms | 5.73 ms | +0.65 ms (~13%) |
| 1,000,000 | 50.3 ms | 58.0 ms | +7.7 ms (~15%) |

The overhead is **pool-bounded**: MMR only considers the top `max(20·k, 100)`
candidates by relevance, so cost does not grow with `N` the way the base matvec
does. The larger 1M figure is mostly cache-miss row gathers over the 1 GB matrix,
not extra compute. In absolute terms it stays well under 10 ms even at 1M chunks.

## Effect on real data (clickhouse-tests)

Query *"clickhouse settings variants and presets"* (top-6):

```
plain (diversity=0)                     diversity=0.4
1 presets.go                            1 presets.go                (0.818)
2 reports/order-u-prewhere-2x2/meta.json 2 ch/client.go             (0.794)
3 reports/agg-dcache-comparison/meta.json 3 reports/100k.../meta.json (0.783)
4 ch/client.go                          4 .gitignore                (0.781)
5 reports/100k-users-...-rate1000/meta  5 go.sum                    (0.772)
6 reports/100k-users-...-rate300/meta   6 docker-compose.yml        (0.763)
```

Plain top-6 was five `report …/meta.json` plus `presets.go`. With diversity it
spreads across source, config and docs — **but also drags in `.gitignore` and
`go.sum`**, which are irrelevant to the question. That is the trade: on a small,
heterogeneous corpus "distinct" often means "unrelated".

## Recommendation

- Keep `diversity=0` as the default (unchanged behavior).
- Calibration: standard MMR's relevance weight `λ` maps to our penalty as
  `diversity = (1-λ)/λ`, so the common library default (`λ=0.5`, e.g. LangChain
  `lambda_mult`) is ranking-equivalent to `diversity ≈ 1.0` (balanced), and
  `λ=0.7` to `diversity ≈ 0.4` (relevance-leaning).
- Start at the **low end** (`0.2–0.4`) and pair with a `min_score` floor; even
  `0.4` pulled unrelated files (`.gitignore`, `go.sum`) on this tiny corpus, so
  small/heterogeneous repos should stay nearer `0.2`.
- It's most useful for **candidate generation** (surfacing the codebase's
  vocabulary from a broad question) where you want distinct files, not five
  near-identical reports.

## Reproduce

```bash
zemsearch bench --sizes 10000,100000,1000000 --diversity 0.3
zemsearch search "clickhouse settings variants and presets" -k 6 \
    --root ~/projects/clickhouse-tests --diversity 0.4
```
