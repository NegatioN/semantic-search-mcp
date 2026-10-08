"""Synthetic search-scaling microbenchmark.

Times the real :class:`NumpyVectorStore` search path over corpus sizes you ask
for, so you can see roughly how latency and memory scale with the number of
chunks. No index, model, or repository is required — records and query vectors
are generated locally.

This is an indicator, not a benchmark suite: it isolates the vector-search cost
and does not include query embedding (a roughly constant HTTP round-trip + model
forward pass) or disk I/O.
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .index.vectors import NumpyVectorStore


def cpu_summary() -> str:
    """Best-effort human-readable CPU description for benchmark output."""
    model = platform.processor() or platform.machine() or "unknown CPU"
    logical = os.cpu_count() or 0
    physical = logical
    try:
        info = Path("/proc/cpuinfo").read_text()
        for line in info.splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
        cores = set()
        for block in info.split("\n\n"):
            phys = core = None
            for line in block.splitlines():
                if line.startswith("physical id"):
                    phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
            if phys is not None and core is not None:
                cores.add((phys, core))
        if cores:
            physical = len(cores)
    except OSError:
        pass
    if physical and physical != logical:
        return f"{model} ({physical} cores / {logical} threads)"
    return f"{model} ({logical} cores)"


@dataclass
class BenchRow:
    n: int
    dim: int
    memory_mb: float
    build_ms: float
    search_ms: float
    filtered_ms: float
    diversified_ms: float
    searches_per_sec: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "dim": self.dim,
            "memory_mb": round(self.memory_mb, 2),
            "build_ms": round(self.build_ms, 3),
            "search_ms": round(self.search_ms, 3),
            "filtered_ms": round(self.filtered_ms, 3),
            "diversified_ms": round(self.diversified_ms, 3),
            "searches_per_sec": round(self.searches_per_sec, 1),
        }


def _synthetic_records(n: int, dim: int, rng: np.random.Generator, filter_ratio: float):
    matrix = rng.standard_normal((n, dim), dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    cutoff = round(filter_ratio * 100)
    records = []
    for i in range(n):
        path = (
            f"src/mod{i % 50}/file{i}.py"
            if (i % 100) < cutoff
            else f"vendor/pkg{i % 50}/file{i}.py"
        )
        records.append(
            {
                "chunk_id": i,
                "path": path,
                "granularity": "file",
                "symbol": None,
                "kind": None,
                "language": "python",
                "start_line": 1,
                "end_line": 1,
                "content": "x" * 200,
                "vector": matrix[i],
            }
        )
    return records


def run_bench(
    sizes: list[int],
    dim: int,
    *,
    repeats: int = 20,
    k: int = 10,
    filter_path: str = "src/",
    filter_ratio: float = 0.1,
    diversity: float = 0.3,
    seed: int = 0,
) -> list[BenchRow]:
    rng = np.random.default_rng(seed)
    rows: list[BenchRow] = []

    for n in sizes:
        records = _synthetic_records(n, dim, rng, filter_ratio)
        store = NumpyVectorStore()

        t0 = time.perf_counter()
        store.build(records)
        build_ms = (time.perf_counter() - t0) * 1000.0

        query = rng.standard_normal(dim, dtype=np.float32)
        query /= np.linalg.norm(query)

        store.search(query, k=k)  # warm up
        reps = repeats if n <= 200_000 else max(3, repeats // 4)

        t0 = time.perf_counter()
        for _ in range(reps):
            store.search(query, k=k)
        search_ms = (time.perf_counter() - t0) * 1000.0 / reps

        store.search(query, k=k, path=filter_path)  # warm up
        t0 = time.perf_counter()
        for _ in range(reps):
            store.search(query, k=k, path=filter_path)
        filtered_ms = (time.perf_counter() - t0) * 1000.0 / reps

        store.search(query, k=k, diversity=diversity)  # warm up
        t0 = time.perf_counter()
        for _ in range(reps):
            store.search(query, k=k, diversity=diversity)
        diversified_ms = (time.perf_counter() - t0) * 1000.0 / reps

        rows.append(
            BenchRow(
                n=n,
                dim=dim,
                memory_mb=n * dim * 4 / 1e6,
                build_ms=build_ms,
                search_ms=search_ms,
                filtered_ms=filtered_ms,
                diversified_ms=diversified_ms,
                searches_per_sec=1000.0 / search_ms if search_ms else 0.0,
            )
        )
        del records, store

    return rows


def format_table(rows: list[BenchRow]) -> str:
    header = (
        f"{'chunks':>10} {'dim':>4} {'mem(MB)':>9} {'build(ms)':>10} "
        f"{'search(ms)':>11} {'filter+(ms)':>12} {'div(ms)':>8} {'search/s':>9}"
    )
    lines = [header, "-" * 80]
    for r in rows:
        lines.append(
            f"{r.n:>10,} {r.dim:>4} {r.memory_mb:>9.1f} {r.build_ms:>10.2f} "
            f"{r.search_ms:>11.3f} {r.filtered_ms:>12.3f} {r.diversified_ms:>8.3f} "
            f"{r.searches_per_sec:>9.0f}"
        )
    return "\n".join(lines)


def rows_to_list(rows: list[BenchRow]) -> list[dict]:
    return [asdict(r) for r in rows]
