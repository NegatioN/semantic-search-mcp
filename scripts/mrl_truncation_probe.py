#!/usr/bin/env python3
"""Probe: persist 768-d vectors and do MRL truncation at query time.

Compares three ways to answer a `dim=d` query when the full 768-d vectors are
available, measuring pure search time and memory:

  A) baseline   : persist (n, d) truncated+normalized vectors; search = A @ q
  B) query-time : persist (n, 768); slice [:, :d] at query time, score =
                  (V[:, :d] @ q) / row_norm, with row norms cached per dim
  C) query-time : same as B but recompute stored row norms on every query

Also verifies B/C return the same ranking as A on the real repo vectors.

Run:  .venv/bin/python scripts/mrl_truncation_probe.py
"""

from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gemma_embedder.index.store import SQLiteStore
from gemma_embedder.model.client import l2_normalize

REAL_STORE = "/tmp/opencode/ch-768/index.db"
SIZES = [10_000, 100_000, 300_000]
DIMS = [128, 256, 512, 768]
REPEATS = 20
K = 10


def load_full(store_path: str) -> np.ndarray:
    with SQLiteStore(store_path) as store:
        records = store.load_records()
    return np.vstack([r["vector"] for r in records]).astype(np.float32)


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def topk(scores: np.ndarray, k: int = K) -> np.ndarray:
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def time_search(fn, repeats: int = REPEATS) -> float:
    fn()  # warm
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) * 1000.0 / repeats


def _matvec(matrix: np.ndarray, q: np.ndarray) -> np.ndarray:
    return matrix @ q


def _score_cached(sub: np.ndarray, q: np.ndarray, rown: np.ndarray) -> np.ndarray:
    return (sub @ q) / rown


def _score_recompute(sub: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (sub @ q) / np.maximum(np.linalg.norm(sub, axis=1), 1e-12)


def make_lazy_cached_search(full: np.ndarray, q: np.ndarray, d: int):
    """Store 768; materialize + cache the normalized (n, d) matrix on first use.

    Returns a zero-arg callable whose first invocation includes the one-time
    build, and whose later invocations are pure contiguous matvec (strategy A).
    """
    cache: dict[int, np.ndarray] = {}

    def run() -> np.ndarray:
        matrix = cache.get(d)
        if matrix is None:
            matrix = normalized_rows(full[:, :d])
            cache[d] = matrix
        return matrix @ q

    return run


def correctness(full: np.ndarray, d: int) -> None:
    q = l2_normalize(full[3, :d].tolist())
    baseline = normalized_rows(full[:, :d])
    query_time = (full[:, :d] @ q) / np.maximum(np.linalg.norm(full[:, :d], axis=1), 1e-12)
    same = np.array_equal(topk(baseline @ q), topk(query_time))
    print(f"  correctness dim={d:>3}: baseline vs query-time top-{K} identical = {same}")


def scaling(d: int) -> None:
    print(f"\nsearch time for dim={d} (ms)")
    print(
        f"  {'n':>9} {'A':>8} {'B':>8} {'C':>8} {'D cold':>9} {'D warm':>8} "
        f"{'mem A':>7} {'mem D':>7}"
    )
    rng = np.random.default_rng(0)
    for n in SIZES:
        full = rng.standard_normal((n, 768), dtype=np.float32)
        full = normalized_rows(full)
        q768 = normalized_rows(rng.standard_normal((1, 768), dtype=np.float32))[0]
        q = l2_normalize(q768[:d].tolist())

        sub = full[:, :d]
        rown = np.linalg.norm(sub, axis=1)

        a_matrix = normalized_rows(sub)
        t_a = time_search(partial(_matvec, a_matrix, q))
        t_b = time_search(partial(_score_cached, sub, q, rown))
        t_c = time_search(partial(_score_recompute, sub, q))

        lazy = make_lazy_cached_search(full, q, d)
        start = time.perf_counter()
        lazy()  # cold: includes materialize + normalize
        t_d_cold = (time.perf_counter() - start) * 1000.0
        t_d_warm = time_search(lazy)  # cached: pure matvec

        mem_a = n * d * 4 / 1e6
        mem_d = n * 768 * 4 / 1e6 + n * d * 4 / 1e6
        print(
            f"  {n:>9,} {t_a:>8.3f} {t_b:>8.3f} {t_c:>8.3f} "
            f"{t_d_cold:>9.3f} {t_d_warm:>8.3f} {mem_a:>6.0f}M {mem_d:>6.0f}M"
        )
        del full, sub, a_matrix, lazy


def main() -> None:
    real = load_full(REAL_STORE)
    print(f"real vectors: {real.shape} (dim 768, stored as-is)")
    print("correctness on real vectors:")
    for d in DIMS:
        correctness(real, d)
    for d in DIMS:
        scaling(d)


if __name__ == "__main__":
    main()
