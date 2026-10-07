#!/usr/bin/env python3
"""Probe: how does ranking change between query_dim=256 and query_dim=768?

Stores are native 768, so this loads the SAME stored vectors at two view
dimensions and compares rankings for a set of architectural queries.

Run:  GEMMA_EMBEDDER_BINARY=/path/llama-server .venv/bin/python scripts/dim_ranking_probe.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gemma_embedder import config as config_mod
from gemma_embedder.model.client import view
from gemma_embedder.model.prefixes import format_query
from gemma_embedder.runtime import Runtime

ROOT = Path.home() / "projects" / "clickhouse-tests"
DIMS = [256, 768]
K = 10

QUERIES = [
    "where is the clickhouse client connection handled",
    "how is the benchmark workload generated for simulated users",
    "clickhouse settings variants and presets",
    "produce a benchmark report summary and percentiles",
    "docker compose services for clickhouse",
    "how does the measurement sampling work",
    "logic that compares variants and picks the best",
    "how are results persisted and read back from reports",
]


def find_binary() -> str:
    env = os.environ.get("GEMMA_EMBEDDER_BINARY")
    if env:
        return env
    for candidate in (
        Path.home() / ".local/share/gemma-embedder/llama.cpp-master/build-cuda/bin/llama-server",
        Path.home() / ".local/share/gemma-embedder/llama.cpp-master/build-static/bin/llama-server",
    ):
        if candidate.exists():
            return str(candidate)
    found = shutil.which("llama-server")
    if not found:
        raise SystemExit("no llama-server found; set GEMMA_EMBEDDER_BINARY")
    return found


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    cfg = config_mod.load(root=ROOT)
    cfg.model.binary = find_binary()
    runtime = Runtime(cfg)
    try:
        runtime.open_store()
        client = runtime.start_model()
        native_q = {q: client.embed_one(format_query(q)) for q in QUERIES}

        rankings: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        paths: list[str] = []
        for d in DIMS:
            cfg.model.query_dim = d
            runtime.reload()
            matrix = runtime.vector._matrix
            paths = [r["path"] for r in runtime.vector._records]
            per_query = {}
            for q in QUERIES:
                qv = np.asarray(view(native_q[q], d), dtype=np.float32)
                scores = matrix @ qv
                per_query[q] = (np.argsort(-scores), scores)
            rankings[d] = per_query
            print(f"dim={d}: snapshot {matrix.shape}")

        lo, hi = DIMS[0], DIMS[-1]
        print(f"\ncomparing query_dim={lo} vs query_dim={hi}, k={K}")
        print(f"{'query':<50} {'top1':>4} {'ov@5':>5} {'ov@10':>5} {'spearman':>9}")
        print("-" * 82)
        top1_changed = 0
        for q in QUERIES:
            o_lo, s_lo = rankings[lo][q]
            o_hi, s_hi = rankings[hi][q]
            same = o_lo[0] == o_hi[0]
            top1_changed += 0 if same else 1
            ov5 = len(set(o_lo[:5]) & set(o_hi[:5])) / 5
            ov10 = len(set(o_lo[:K]) & set(o_hi[:K])) / K
            rho = spearman(s_lo, s_hi)
            print(f"{q:<50} {('yes' if same else 'NO'):>4} {ov5:>5.1f} {ov10:>5.1f} {rho:>9.3f}")
        print("-" * 82)
        print(f"top-1 changed on {top1_changed}/{len(QUERIES)} queries")

        print("\ntop-3 detail:")
        for q in QUERIES:
            print(f"\n  Q: {q}")
            for d in DIMS:
                o, s = rankings[d][q]
                top = ", ".join(f"{paths[i]}({s[i]:.3f})" for i in o[:3])
                print(f"    {d:>3}d: {top}")
    finally:
        runtime.close()


if __name__ == "__main__":
    sys.exit(main())
