#!/usr/bin/env python3
"""Estimate how many embeddings a CodeGraph database would produce, and the
approximate query latency against that many vectors.

The count mirrors exactly what ``CodeGraphIndexer`` embeds: one embedding per
symbol node (function/method/struct/type_alias/constant), deduplicated by
``(file_path, qualified_name, kind)``.

Query latency is the in-memory vector-scan cost, measured on this machine with
the same ``NumpyVectorStore`` path search uses. It excludes the query-embedding
model call, which is a roughly constant cost per query regardless of index size.

Usage:
    python scripts/codegraph_embed_count.py [PATH] [--dim 256] [-k 10]

PATH may be the ``codegraph.db`` file, the ``.codegraph`` directory, or a repo
root (defaults to ``./.codegraph/codegraph.db``).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zemsearch.bench import cpu_summary, run_bench
from zemsearch.codegraph import SYMBOL_KINDS


def resolve_db(raw: str) -> Path:
    p = Path(raw)
    if p.is_dir():
        p = p / "codegraph.db" if p.name == ".codegraph" else p / ".codegraph/codegraph.db"
    return p


def count_embeddings(db_path: Path) -> int:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        placeholders = ",".join("?" * len(SYMBOL_KINDS))
        row = conn.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT DISTINCT file_path, qualified_name, kind FROM nodes"
            f"  WHERE kind IN ({placeholders})"
            ")",
            tuple(sorted(SYMBOL_KINDS)),
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", nargs="?", default=".codegraph/codegraph.db")
    ap.add_argument("--dim", type=int, default=256, help="query dimension (MRL view)")
    ap.add_argument("-k", type=int, default=10, help="results per query")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument(
        "--max-measure",
        type=int,
        default=1_000_000,
        help="cap the measured corpus size and scale linearly above it",
    )
    args = ap.parse_args()

    db = resolve_db(args.path)
    if not db.is_file():
        print(f"error: no such database: {db}", file=sys.stderr)
        return 1

    n = count_embeddings(db)
    print(f"embeddings: {n:,}")
    if n == 0:
        print("query scan: n/a (empty index)")
        return 0

    measured_n = min(n, args.max_measure)
    row = run_bench([measured_n], args.dim, repeats=args.repeats, k=args.k)[0]
    scan_ms = row.search_ms * (n / measured_n)
    scaled = "" if measured_n == n else f" (measured {measured_n:,}, scaled)"
    print(f"query scan: ~{scan_ms:.4f} ms{scaled}")
    print(f"memory:     ~{n * args.dim * 4 / 1e6:,.1f} MB  (dim={args.dim})")
    print("note: scan only; excludes the per-query model call (roughly constant)")
    print(f"cpu: {cpu_summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
