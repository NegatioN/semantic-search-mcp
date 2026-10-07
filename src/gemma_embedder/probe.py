"""Phase 0 probe: prove the real embedding path end-to-end.

Embeds whole files from a workspace with the exact EmbeddingGemma 2 document
prefix, then ranks them for a set of semantic queries using the query prefix.
Also verifies output dimension, unit norm and determinism.

This is intentionally dependency-free (stdlib only) so it can run before the
rest of the project is installed.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

from .model.client import EmbeddingClient, WarmupReport, dot
from .model.prefixes import format_document, format_query
from .model.server import (
    DEFAULT_HOST,
    DEFAULT_MODEL_REPO,
    DEFAULT_PORT,
    INSTALL_HINT,
    ServerManager,
    find_binary,
)

DEFAULT_EXTENSIONS = (".py", ".md", ".toml", ".txt", ".rs", ".ts", ".js", ".go")
DEFAULT_MAX_BYTES = 200_000

#: (query, acceptable top-1 filename hints) pairs. A query passes if the top-1
#: path contains any of its hints. Hints are soft checks, not ground truth.
DEFAULT_QUERIES: list[tuple[str, tuple[str, ...] | None]] = [
    ("how do we format the task prefix for a semantic code search query", ("prefixes",)),
    ("how do we POST inputs to the embeddings endpoint and parse the response", ("client",)),
    ("how is the llama-server subprocess started and stopped", ("server",)),
    (
        "how to run the embedding model natively with llama.cpp and which flags",
        ("GUIDE", "README"),
    ),
    (
        "what vector similarity metric should be used and why normalization matters",
        ("GUIDE", "README", "similarity"),
    ),
    ("what tools does the MCP server expose for search and reindex", ("GUIDE", "README")),
]


@dataclass
class QueryOutcome:
    query: str
    expected: tuple[str, ...] | None
    ranking: list[tuple[str, float]]
    passed: bool


@dataclass
class ProbeResult:
    warmup: WarmupReport
    root: str
    files: list[str] = field(default_factory=list)
    outcomes: list[QueryOutcome] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.warmup.ok and all(o.passed for o in self.outcomes)


def iter_text_files(
    root: Path,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    limit: int | None = None,
) -> list[Path]:
    """Return sorted text files under ``root`` matching ``extensions``."""
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in extensions:
            continue
        if any(part in {".git", ".venv", "node_modules", "__pycache__"} for part in path.parts):
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        files.append(path)
        if limit is not None and len(files) >= limit:
            break
    return files


def parse_host_port(url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(url)
    return parsed.hostname or DEFAULT_HOST, parsed.port or DEFAULT_PORT


def run_probe(
    client: EmbeddingClient,
    root: Path,
    queries: list[tuple[str, tuple[str, ...] | None]],
    *,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    limit: int | None = None,
    top_k: int = 3,
) -> ProbeResult:
    """Embed files under ``root`` and rank them for each query."""
    warmup = client.warmup()
    paths = iter_text_files(root, extensions, limit=limit)
    titles = [str(p.relative_to(root)) for p in paths]

    documents: list[str] = []
    kept: list[str] = []
    for path, title in zip(paths, titles):
        try:
            code = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        documents.append(format_document(code, title=title))
        kept.append(title)

    vectors = client.embed(documents) if documents else []

    outcomes: list[QueryOutcome] = []
    for query, expected in queries:
        if not vectors:
            outcomes.append(QueryOutcome(query, expected, [], False))
            continue
        qvec = client.embed_one(format_query(query))
        scored = sorted(
            ((path, dot(qvec, vec)) for path, vec in zip(kept, vectors)),
            key=lambda item: item[1],
            reverse=True,
        )
        top = scored[:top_k]
        passed = (
            bool(top)
            and expected is not None
            and any(hint in top[0][0] for hint in expected)
        )
        outcomes.append(QueryOutcome(query, expected, top, passed))

    return ProbeResult(warmup=warmup, root=str(root), files=kept, outcomes=outcomes)


def print_report(result: ProbeResult) -> None:
    w = result.warmup
    print("=" * 72)
    print("Phase 0 probe — EmbeddingGemma 2 via llama.cpp")
    print("=" * 72)
    print(f"root           : {result.root}")
    print(f"files embedded : {len(result.files)}")
    print(f"dimension      : {w.dim} (native 768)")
    print(f"unit norm      : {w.norm:.6f}")
    print(f"deterministic  : {w.deterministic}")
    print(f"warmup ok      : {w.ok}")
    print("-" * 72)
    for outcome in result.outcomes:
        status = "PASS" if outcome.passed else "FAIL"
        print(f"[{status}] {outcome.query}")
        if outcome.expected:
            print(f"         acceptable top-1 hints: {', '.join(outcome.expected)}")
        for rank, (path, score) in enumerate(outcome.ranking, 1):
            print(f"         {rank}. {score:+.4f}  {path}")
        print()
    print("-" * 72)
    print("RESULT:", "PASS" if result.passed else "FAIL")
    print("=" * 72)


def result_to_dict(result: ProbeResult) -> dict:
    return {
        "root": result.root,
        "files": result.files,
        "warmup": {
            "dim": result.warmup.dim,
            "norm": result.warmup.norm,
            "deterministic": result.warmup.deterministic,
            "ok": result.warmup.ok,
        },
        "queries": [
            {
                "query": o.query,
                "expected": list(o.expected) if o.expected else None,
                "passed": o.passed,
                "ranking": [{"path": p, "score": s} for p, s in o.ranking],
            }
            for o in result.outcomes
        ],
        "passed": result.passed,
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="directory to index for the probe (default: project root)",
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--binary", default=None, help="path to llama/llama-server binary")
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--dim", type=int, default=0, help="MRL truncation (0 = native 768)")
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="do not start a server; require an already-running one",
    )
    parser.add_argument("--limit", type=int, default=None, help="max files to embed")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--query", action="append", default=None, help="override default queries")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")


def run_args(args: argparse.Namespace) -> int:
    host, port = parse_host_port(args.server_url)
    binary = None if args.no_spawn else find_binary(args.binary)

    manager = ServerManager(
        binary,
        model_repo=args.model_repo,
        host=host,
        port=0 if binary else port,
        gpu_layers=args.gpu_layers,
    )
    if binary:
        print(f"Starting server: {binary} (port {host}:auto)")
        manager.start()
    server_url = manager.base_url if binary else args.server_url
    if binary:
        print(f"Server ready at {server_url}")

    try:
        client = EmbeddingClient(server_url, dim=args.dim or None)
        if not client.health():
            print(f"ERROR: embeddings server is not healthy at {server_url}.")
            if not binary:
                print(f"Install llama.cpp:   {INSTALL_HINT}")
                print(
                    "Then start it:       "
                    f"llama serve -hf {args.model_repo} --embeddings "
                    "--pooling mean --embd-normalize 2 --ctx-size 8192"
                )
                print("Or point --server-url at an already-running server.")
            return 2

        queries = (
            [(q, None) for q in args.query] if args.query else list(DEFAULT_QUERIES)
        )
        result = run_probe(
            client,
            args.root,
            queries,
            limit=args.limit,
            top_k=args.top_k,
        )
        if args.json:
            print(json.dumps(result_to_dict(result), indent=2))
        else:
            print_report(result)
        return 0 if result.passed else 1
    finally:
        manager.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="probe_model", description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)
    return run_args(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
