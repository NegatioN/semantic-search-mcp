#!/usr/bin/env python3
"""Probe: does ranking change if we use unnormalized embeddings with dot product?

Runs a llama-server with ``--embd-normalize -1`` (raw, NON-unit vectors), embeds
the clickhouse-tests corpus, then ranks a set of queries two ways over the *same*
vectors:

  - cosine : normalize each doc and the query (L2), then dot product
  - dot    : raw dot product on the unnormalized vectors

Because the server is asked for raw vectors, the cosine arm reproduces exactly
what the production pipeline does (server-normalized 768 -> truncate d ->
renormalize), so any difference is purely the normalization effect.

Run:  ZEMSEARCH_BINARY=/path/llama-server .venv/bin/python scripts/unnormalized_probe.py
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

from zemsearch import config as config_mod
from zemsearch.index.walker import walk
from zemsearch.model.prefixes import format_document, format_query

ROOT = Path.home() / "projects" / "clickhouse-tests"
MODEL_REPO = "ggml-org/embeddinggemma-2-GGUF:Q8_0"
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
]


def find_binary() -> str:
    env = os.environ.get("ZEMSEARCH_BINARY")
    if env:
        return env
    for candidate in (
        Path.home() / ".local/share/zemsearch/llama.cpp-master/build-cuda/bin/llama-server",
        Path.home() / ".local/share/zemsearch/llama.cpp-master/build-static/bin/llama-server",
    ):
        if candidate.exists():
            return str(candidate)
    found = shutil.which("llama-server")
    if not found:
        raise SystemExit("no llama-server found; set ZEMSEARCH_BINARY")
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
                body = json.loads(resp.read())
            if body.get("status") in ("ok", True):
                return
        except Exception:  # noqa: BLE001, S110 - expected until the server is ready
            pass
        time.sleep(1.0)
    raise SystemExit("server did not become healthy")


def embed(base: str, texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i : i + BATCH]
        payload = json.dumps({"model": "embeddinggemma-2", "input": chunk}).encode()
        req = urllib.request.Request(
            f"{base}/v1/embeddings", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())["data"]
        vectors.extend(item["embedding"] for item in sorted(data, key=lambda x: x["index"]))
    return np.asarray(vectors, dtype=np.float32)


def l2(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-12)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"missing {ROOT}")
    cfg = config_mod.load(root=ROOT).index
    files = walk(ROOT.resolve(), cfg)[:MAX_FILES]
    titles = [f.relative_to(ROOT).as_posix() for f in files]
    documents = [
        format_document(f.read_text(encoding="utf-8", errors="replace"), title=t)
        for f, t in zip(files, titles)
    ]
    print(f"corpus: {len(documents)} files from {ROOT}")

    binary = find_binary()
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    cmd = [
        binary,
        "-hf",
        MODEL_REPO,
        "--embeddings",
        "--pooling",
        "mean",
        "--embd-normalize",
        "-1",
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
    print(f"starting (raw/unnormalized): {binary}")
    proc = subprocess.Popen(cmd, stdout=sys.stderr, stderr=sys.stderr)
    try:
        wait_health(base)
        raw_docs = embed(base, documents)[:, :DIM]  # unnormalized, truncated
        raw_q = embed(base, [format_query(q) for q in QUERIES])[:, :DIM]

        doc_norms = np.linalg.norm(raw_docs, axis=1)
        print(
            f"\nraw {DIM}-d vector norms: min={doc_norms.min():.3f} "
            f"median={np.median(doc_norms):.3f} max={doc_norms.max():.3f} "
            f"(spread drives dot-vs-cosine drift)"
        )

        cos_docs, cos_q = l2(raw_docs), l2(raw_q)
        print(f"\n{'query':<52} {'top1 same':>9} {'ov@5':>5} {'spearman':>9}")
        print("-" * 80)
        total_ov, changed = 0.0, 0
        for qi, query in enumerate(QUERIES):
            cosine = cos_docs @ cos_q[qi]
            dot = raw_docs @ raw_q[qi]
            cos_top = np.argsort(-cosine)[:5]
            dot_top = np.argsort(-dot)[:5]
            overlap = len(set(cos_top.tolist()) & set(dot_top.tolist())) / 5
            same_top1 = cos_top[0] == dot_top[0]
            changed += 0 if same_top1 else 1
            total_ov += overlap
            rho = spearman(cosine, dot)
            print(f"{query:<52} {same_top1!s:>9} {overlap:>5.1f} {rho:>9.3f}")
            if not same_top1:
                print(f"    cosine top1: {titles[cos_top[0]]}  |  dot top1: {titles[dot_top[0]]}")
        print("-" * 80)
        print(
            f"top1 changed on {changed}/{len(QUERIES)} queries; "
            f"mean overlap@5 = {total_ov / len(QUERIES):.2f}; "
            f"min spearman = {min(spearman(cos_docs @ cos_q[i], raw_docs @ raw_q[i]) for i in range(len(QUERIES))):.3f}"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
