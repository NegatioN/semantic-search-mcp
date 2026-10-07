#!/usr/bin/env python3
"""Probe: load only the leading `d` dims from disk instead of the full 768.

Strategy D keeps the full (n, 768) vectors in RAM and materializes a normalized
(n, d) cache. Strategy E reads only the first `d` floats of each stored vector
(``substr(vector, 1, d*4)``) when the target dimension is known up front, so the
in-memory footprint is just (n, d) — the 768 copy never exists.

Measures load time, peak Python/numpy allocation, and bytes read from SQLite.

Run:  .venv/bin/python scripts/mrl_lazy_load_probe.py
"""

from __future__ import annotations

import sqlite3
import sys
import time
import tracemalloc
from functools import partial
from pathlib import Path

import numpy as np

SIZES = [20_000, 100_000]
DIMS = [128, 256, 512]
DB = Path("/tmp/opencode/mrl-lazy-load.db")


def build_store(n: int, dim: int = 768, seed: int = 0) -> None:
    if DB.exists():
        DB.unlink()
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n, dim), dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE embeddings (chunk_id INTEGER PRIMARY KEY, dim INTEGER, vector BLOB)")
    conn.executemany(
        "INSERT INTO embeddings(chunk_id, dim, vector) VALUES(?, ?, ?)",
        ((i, dim, matrix[i].tobytes()) for i in range(n)),
    )
    conn.commit()
    conn.close()


def full_load(conn: sqlite3.Connection) -> tuple[np.ndarray, int]:
    rows = conn.execute("SELECT vector FROM embeddings").fetchall()
    matrix = np.vstack([np.frombuffer(r[0], dtype="<f4") for r in rows])
    return matrix, sum(len(r[0]) for r in rows)


def prefix_load(conn: sqlite3.Connection, d: int) -> tuple[np.ndarray, int]:
    rows = conn.execute("SELECT substr(vector, 1, ?) FROM embeddings", (d * 4,)).fetchall()
    matrix = np.vstack([np.frombuffer(r[0], dtype="<f4") for r in rows])
    return matrix, sum(len(r[0]) for r in rows)


def measure(fn):
    tracemalloc.start()
    start = time.perf_counter()
    result = fn()
    elapsed = (time.perf_counter() - start) * 1000.0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


def check_correctness() -> None:
    build_store(64)
    conn = sqlite3.connect(DB)
    full, _ = full_load(conn)
    for d in DIMS + [768]:
        prefix, _ = prefix_load(conn, d)
        assert np.array_equal(full[:, :d], prefix), f"mismatch at d={d}"
    conn.close()
    print("correctness: prefix load == full[:, :d] for d in {128,256,512,768} (asserted)")


def main() -> None:
    check_correctness()
    print(f"\ndb: {DB}")
    for n in SIZES:
        build_store(n)
        conn = sqlite3.connect(DB)
        (full_mat, full_bytes), t_full, peak_full = measure(partial(full_load, conn))
        print(f"\nn={n:,}  (768-d blobs on disk: {n * 768 * 4 / 1e6:.0f} MB)")
        print(
            f"  full load        : peak {peak_full / 1e6:6.1f} MB   {t_full:6.0f} ms   "
            f"read {full_bytes / 1e6:6.0f} MB"
        )
        del full_mat
        for d in DIMS:
            (prefix, prefix_bytes), t_pre, peak_pre = measure(partial(prefix_load, conn, d))
            print(
                f"  prefix load d={d:<4}: peak {peak_pre / 1e6:6.1f} MB   {t_pre:6.0f} ms   "
                f"read {prefix_bytes / 1e6:6.0f} MB"
            )
            del prefix
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
