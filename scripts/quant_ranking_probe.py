#!/usr/bin/env python3
"""Probe: does Q8_0 vs BF16 change embeddings or rankings?

Serves the same repo twice (Q8_0, then BF16), embeds the clickhouse-tests corpus
and a set of queries with each, and compares:

  - vector-level agreement (cosine between the two quants' doc embeddings)
  - ranking agreement at query_dim=256 (top-1 / overlap@k / Spearman)

Run:  GEMMA_EMBEDDER_BINARY=/path/llama-server .venv/bin/python scripts/quant_ranking_probe.py
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gemma_embedder import config as config_mod
from gemma_embedder.index.walker import walk
from gemma_embedder.model.client import l2_normalize
from gemma_embedder.model.prefixes import format_document, format_query

ROOT = Path.home() / "projects" / "clickhouse-tests"
REPO = "ggml-org/embeddinggemma-2-GGUF"
QUANTS = ["Q8_0", "BF16"]
DIM = 256
BATCH = 16
MAX_FILES = 200

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
    for c in (
        Path.home() / ".local/share/gemma-embedder/llama.cpp-master/build-cuda/bin/llama-server",
        Path.home() / ".local/share/gemma-embedder/llama.cpp-master/build-static/bin/llama-server",
    ):
        if c.exists():
            return str(c)
    found = shutil.which("llama-server")
    if not found:
        raise SystemExit("no llama-server found; set GEMMA_EMBEDDER_BINARY")
    return found


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_health(base: str, timeout: float = 600.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as resp:
                if json.loads(resp.read()).get("status") in ("ok", True):
                    return
        except Exception:  # noqa: BLE001, S110 - expected until ready
            pass
        time.sleep(1.0)
    raise SystemExit("server did not become healthy")


def embed(base: str, texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        payload = json.dumps({"model": "embeddinggemma-2", "input": texts[i : i + BATCH]}).encode()
        req = urllib.request.Request(
            f"{base}/v1/embeddings", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())["data"]
        vectors.extend(x["embedding"] for x in sorted(data, key=lambda x: x["index"]))
    return np.asarray(vectors, dtype=np.float32)


def with_quant(binary: str, quant: str, documents: list[str], queries: list[str]):
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    cmd = [
        binary,
        "-hf",
        f"{REPO}:{quant}",
        "--embeddings",
        "--no-mmproj",
        "--pooling",
        "mean",
        "--embd-normalize",
        "2",
        "--ctx-size",
        "8192",
        "--batch-size",
        "8192",
        "--ubatch-size",
        "8192",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    print(f"serving {quant} ...")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_health(base)
        docs = embed(base, documents)
        qs = embed(base, [format_query(q) for q in queries])
        return docs, qs
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()


def view_l2(matrix: np.ndarray, d: int) -> np.ndarray:
    sub = matrix[:, :d]
    return sub / np.maximum(np.linalg.norm(sub, axis=1, keepdims=True), 1e-12)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    cfg = config_mod.load(root=ROOT).index
    files = walk(ROOT.resolve(), cfg)[:MAX_FILES]
    titles = [f.relative_to(ROOT).as_posix() for f in files]
    documents = [
        format_document(f.read_text(encoding="utf-8", errors="replace"), title=t)
        for f, t in zip(files, titles)
    ]
    print(f"corpus: {len(documents)} files")

    binary = find_binary()
    data = {}
    for quant in QUANTS:
        data[quant] = with_quant(binary, quant, documents, QUERIES)

    q8_docs, bf_docs = data["Q8_0"][0], data["BF16"][0]

    # --- vector-level agreement ---
    cos = np.sum(l2_normalize_rows(q8_docs) * l2_normalize_rows(bf_docs), axis=1)
    maxdiff = np.abs(q8_docs - bf_docs).max()
    print(
        f"\ndoc-vector agreement Q8_0 vs BF16 (native 768): "
        f"cosine min={cos.min():.5f} mean={cos.mean():.5f} max={cos.max():.5f}; "
        f"max |Δelement|={maxdiff:.3e}"
    )

    # --- ranking agreement at query_dim ---
    d = DIM
    A = view_l2(q8_docs, d)
    B = view_l2(bf_docs, d)
    print(f"\nranking agreement at query_dim={d}")
    print(f"{'query':<50} {'top1':>4} {'ov@5':>5} {'ov@10':>5} {'spearman':>9}")
    print("-" * 82)
    changed = 0
    for i, query in enumerate(QUERIES):
        qa = l2_normalize(data["Q8_0"][1][i, :d].tolist())
        qb = l2_normalize(data["BF16"][1][i, :d].tolist())
        sa = A @ np.asarray(qa, dtype=np.float32)
        sb = B @ np.asarray(qb, dtype=np.float32)
        oa, ob = np.argsort(-sa), np.argsort(-sb)
        same = oa[0] == ob[0]
        changed += 0 if same else 1
        ov5 = len(set(oa[:5]) & set(ob[:5])) / 5
        ov10 = len(set(oa[:10]) & set(ob[:10])) / 10
        print(
            f"{query:<50} {('yes' if same else 'NO'):>4} {ov5:>5.1f} {ov10:>5.1f} {spearman(sa, sb):>9.3f}"
        )
    print("-" * 82)
    print(f"top-1 changed on {changed}/{len(QUERIES)} queries")


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


if __name__ == "__main__":
    sys.exit(main())
