#!/usr/bin/env python3
"""Probe: embedding throughput, CPU build vs CUDA build of llama-server.

Spawns each server against the same EmbeddingGemma 2 weights, warms up, then
embeds a deterministic corpus in fixed-size batches. Reports wall time, docs/s
and chars/s so the CPU-vs-GPU gap is measurable and reproducible.

Examples:
    python scripts/embed_throughput_probe.py \
        --cpu-binary ~/.local/share/zemsearch/llama.cpp/build-static/bin/llama-server \
        --gpu-binary ~/.local/share/zemsearch/llama.cpp/build-cuda/bin/llama-server
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zemsearch.model.client import EmbeddingClient
from zemsearch.model.server import DEFAULT_MODEL_REPO, ServerManager

_BASE = (
    "def handle(request): return pipeline.authenticate(request).route(handler) "
    "# validate payload, resolve middleware, emit response, record metrics "
)


def make_corpus(count: int, chars: int) -> list[str]:
    """``count`` near-identical documents of ~``chars`` characters each."""
    body = (_BASE * ((chars // len(_BASE)) + 1))[:chars]
    return [f"unit {i}: {body}" for i in range(count)]


def run(binary: str, gpu_layers: int, texts: list[str], batch: int) -> dict:
    mgr = ServerManager(
        binary,
        model_repo=DEFAULT_MODEL_REPO,
        port=0,  # pick a free port; never collide with a pre-existing server
        gpu_layers=gpu_layers,
        health_timeout=600,
    )
    mgr.start()
    try:
        client = EmbeddingClient(mgr.base_url)
        client.embed(texts[:batch])  # warm up (load kernels / caches)
        started = time.perf_counter()
        for i in range(0, len(texts), batch):
            client.embed(texts[i : i + batch])
        elapsed = time.perf_counter() - started
    finally:
        mgr.stop()
    chars = sum(len(t) for t in texts)
    return {
        "docs": len(texts),
        "chars": chars,
        "seconds": elapsed,
        "docs_per_s": len(texts) / elapsed,
        "chars_per_s": chars / elapsed,
    }


def _fmt(label: str, res: dict, baseline: dict | None) -> str:
    speedup = ""
    if baseline:
        speedup = f"  ({res['docs_per_s'] / baseline['docs_per_s']:.1f}x)"
    return (
        f"{label:<10} {res['seconds']:>8.2f}s  "
        f"{res['docs_per_s']:>8.1f} docs/s  {res['chars_per_s'] / 1000:>8.1f} kchar/s{speedup}"
    )


def main() -> int:
    home = Path.home()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cpu-binary",
        default=str(home / ".local/share/zemsearch/llama.cpp/build-static/bin/llama-server"),
    )
    ap.add_argument(
        "--gpu-binary",
        default=str(home / ".local/share/zemsearch/llama.cpp/build-cuda/bin/llama-server"),
    )
    ap.add_argument("--docs", type=int, default=256)
    ap.add_argument("--chars", type=int, default=800, help="approximate characters per doc")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gpu-layers", type=int, default=99)
    args = ap.parse_args()

    texts = make_corpus(args.docs, args.chars)
    print(f"corpus: {args.docs} docs x ~{args.chars} chars  (batch {args.batch}, BF16)")
    print("-" * 78)

    cpu = None
    if Path(args.cpu_binary).is_file():
        cpu = run(args.cpu_binary, 0, texts, args.batch)
        print(_fmt("CPU", cpu, None))
    else:
        print(f"CPU        skipped (no binary at {args.cpu_binary})")

    if Path(args.gpu_binary).is_file():
        gpu = run(args.gpu_binary, args.gpu_layers, texts, args.batch)
        print(_fmt("GPU", gpu, cpu))
    else:
        print(f"GPU        skipped (no binary at {args.gpu_binary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
