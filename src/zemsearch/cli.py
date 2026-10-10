"""Console entrypoint for zemsearch.

Commands: ``doctor``, ``probe``, ``index``, ``watch``, ``search``, ``status``,
``serve``, ``bench``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

from . import __version__, probe
from . import config as config_mod
from .model.client import EmbeddingClient
from .model.server import INSTALL_HINT, find_binary
from .runtime import Runtime, RuntimeStatus


def _load_config(args: argparse.Namespace):
    cfg = config_mod.load(getattr(args, "config", None), getattr(args, "root", None))
    binary = getattr(args, "binary", None)
    if binary:
        cfg.model.binary = binary
    server_url = getattr(args, "server_url", None)
    if server_url:
        cfg.model.server_url = server_url
    if getattr(args, "no_spawn", False):
        cfg.model.manage_server = False
    return cfg


def _cmd_doctor(args: argparse.Namespace) -> int:
    binary = find_binary(args.binary or os.environ.get("ZEMSEARCH_BINARY"))
    print(f"llama binary : {binary or 'NOT FOUND'}")
    if not binary:
        print(f"install with : {INSTALL_HINT}")

    healthy = EmbeddingClient(args.server_url).health()
    print(f"server       : {args.server_url}")
    print(f"health       : {'ok' if healthy else 'unreachable'}")
    return 0 if (binary or healthy) else 1


def _cmd_index(args: argparse.Namespace) -> int:
    cfg = _load_config(args)

    def progress(path: str, chunks: int) -> None:
        if not args.quiet:
            print(f"  + {path} ({chunks} chunk{'s' if chunks != 1 else ''})")

    with Runtime(cfg) as runtime:
        report = runtime.index(subpath=args.path, force=args.force, progress=progress)
    print(json.dumps(report.as_dict(), indent=2))
    return 0 if not report.errors else 1


def _cmd_watch(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    if not cfg.reindex.watch:
        print("watcher disabled (reindex.watch = false)", file=sys.stderr)
        return 1

    with Runtime(cfg) as runtime:
        runtime.start_model()
        if args.no_initial:
            runtime.reload()
        else:
            report = runtime.index()
            print(
                f"indexed: {report.changed} changed, {report.unchanged} unchanged, "
                f"{report.chunks} chunks",
                file=sys.stderr,
            )
        runtime.start_watching()
        print(
            f"watching for changes (debounce {cfg.reindex.debounce_seconds}s, "
            f"interval {cfg.reindex.interval_seconds}s) — ctrl-c to stop",
            file=sys.stderr,
        )
        try:
            while runtime.watcher is not None and runtime.watcher.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    return 0


def _print_hits(hits) -> None:
    if not hits:
        print("no results")
        return
    for rank, hit in enumerate(hits, 1):
        location = f"{hit.path}:{hit.start_line}-{hit.end_line}"
        symbol = f"  [{hit.symbol}]" if hit.symbol else ""
        print(f"{rank:>2}. {hit.score:+.4f}  {location}{symbol}")
        first = hit.snippet.strip().splitlines()
        if first:
            print(f"      {first[0][:100]}")


def _cmd_search(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    with Runtime(cfg) as runtime:
        hits = runtime.search(
            args.query,
            k=args.k,
            path=args.path,
            min_score=args.min_score,
            granularity=args.granularity,
            diversity=args.diversity,
        )
        status = runtime.status()
    if args.json:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "count": len(hits),
                    "model": status.model_repo,
                    "results": [hit.as_dict() for hit in hits],
                },
                indent=2,
            )
        )
    else:
        _print_hits(hits)
    return 0 if hits else 1


def _print_status(status: RuntimeStatus) -> None:
    print(f"root          : {status.root}")
    print(f"files         : {status.files}")
    print(f"chunks        : {status.chunks}")
    print(f"vectors       : {status.vectors}")
    print(
        f"dimensions    : native={status.native_dim} storage={status.storage_dim} "
        f"query={status.query_dim} loaded={status.loaded_dim}"
    )
    print(f"model         : {status.model_repo}")
    print(f"server        : {status.server_url} (managed={status.server_managed})")
    print(
        f"watching      : {'yes' if status.watching else 'no'} "
        f"(interval {int(status.watch_interval_seconds)}s)"
    )
    print(f"last reindex  : {status.last_reindex}")
    if status.by_language:
        langs = ", ".join(f"{k}={v}" for k, v in sorted(status.by_language.items()))
        print(f"languages     : {langs}")


def _cmd_status(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    with Runtime(cfg) as runtime:
        runtime.reload()
        status = runtime.status()
    if args.json:
        print(json.dumps(status.as_dict(), indent=2))
    else:
        _print_status(status)
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from .bench import cpu_summary, format_table, run_bench

    cfg = _load_config(args)
    dim = args.dim or cfg.model.query_dim
    sizes = [int(s) for s in args.sizes.replace(" ", "").split(",") if s]
    cpu = cpu_summary()
    rows = run_bench(sizes, dim, repeats=args.repeats, k=args.k, diversity=args.diversity)
    if args.json:
        payload = {
            "cpu": cpu,
            "dim": dim,
            "k": args.k,
            "rows": [r.as_dict() for r in rows],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"CPU: {cpu}  (CPU-only; search is memory-bandwidth bound)")
        print(f"Search scaling (synthetic; dim={dim}, k={args.k}, path filter ~10%)")
        print(format_table(rows))
        print()
        print("Notes:")
        print("  search(ms)  = vectorized dot product over all chunks (query embedding excluded)")
        print("  filter+(ms) = same, plus the vectorized path/granularity mask (NumPy/C-level)")
        print("  div(ms)     = same, plus MMR diversity re-ranking (pool is bounded)")
        print("  build(ms)   = snapshot rebuild on every reindex, incl. metadata arrays")
        print("  mem(MB)     = n * dim * 4 for vectors; path/granularity arrays add more")
        print("  threads     = dot product uses OpenBLAS (~1-4 threads, plateaus; not all cores);")
        print("                query embedding runs in llama.cpp (multi-threaded) and is excluded.")
    return 0


def _start_background_index(runtime: Runtime) -> None:
    """Run the initial index off the MCP handshake path."""

    def work() -> None:
        try:
            report = runtime.index()
            print(
                f"initial index: {report.changed} changed, "
                f"{report.unchanged} unchanged, {report.chunks} chunks",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001 - never crash the server
            print(f"initial index failed: {exc}", file=sys.stderr)

    threading.Thread(target=work, name="zemsearch-initial-index", daemon=True).start()


def _cmd_serve(args: argparse.Namespace) -> int:
    from .mcp.server import run as run_server

    cfg = _load_config(args)
    with Runtime(cfg) as runtime:
        runtime.open_store()
        runtime.reload()
        if cfg.reindex.watch:
            runtime.start_watching()
            print("watching for changes", file=sys.stderr)
        if runtime.is_initialized():
            # Only refresh repos that were explicitly initialized.
            if cfg.reindex.update_on_start:
                _start_background_index(runtime)
        else:
            print(
                "no index for this repo yet; call the `reindex` tool to build one",
                file=sys.stderr,
            )
        run_server(runtime, cfg.server.transport)
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="path to zemsearch.toml")
    parser.add_argument("--root", default=None, help="workspace root (default: cwd)")
    parser.add_argument("--binary", default=None, help="path to llama/llama-server")
    parser.add_argument("--server-url", default=None)
    parser.add_argument(
        "--no-spawn", action="store_true", help="use an external server, do not spawn"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zemsearch")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check for a llama binary and running server")
    doctor.add_argument("--binary", default=None)
    doctor.add_argument("--server-url", default="http://127.0.0.1:8080")
    doctor.set_defaults(func=_cmd_doctor)

    probe_parser = sub.add_parser("probe", help="run the Phase 0 end-to-end probe")
    probe.add_arguments(probe_parser)
    probe_parser.set_defaults(func=probe.run_args)

    index = sub.add_parser("index", help="index or refresh the workspace")
    _add_common(index)
    index.add_argument("--path", default=None, help="subpath to index")
    index.add_argument("--force", action="store_true", help="re-embed everything")
    index.add_argument("--quiet", action="store_true")
    index.set_defaults(func=_cmd_index)

    watch = sub.add_parser("watch", help="index and re-index on file changes")
    _add_common(watch)
    watch.add_argument("--no-initial", action="store_true", help="skip the initial index pass")
    watch.set_defaults(func=_cmd_watch)

    search = sub.add_parser("search", help="semantic search the index")
    _add_common(search)
    search.add_argument("query")
    search.add_argument("-k", type=int, default=10)
    search.add_argument("--path", default=None)
    search.add_argument("--min-score", type=float, default=0.0)
    search.add_argument("--granularity", choices=["any", "file", "symbol"], default="any")
    search.add_argument(
        "--diversity",
        type=float,
        default=0.0,
        help="MMR strength in [0,1]; 0 = plain top-k (default)",
    )
    search.add_argument("--json", action="store_true")
    search.set_defaults(func=_cmd_search)

    status = sub.add_parser("status", help="show index status")
    _add_common(status)
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=_cmd_status)

    serve = sub.add_parser("serve", help="run the MCP server (stdio)")
    _add_common(serve)
    serve.set_defaults(func=_cmd_serve)

    bench = sub.add_parser(
        "bench", help="microbenchmark search scaling on synthetic data (no model)"
    )
    bench.add_argument("--config", default=None)
    bench.add_argument("--root", default=None)
    bench.add_argument("--dim", type=int, default=0, help="defaults to the config dim")
    bench.add_argument(
        "--sizes", default="1000,10000,100000,1000000", help="comma-separated chunk counts"
    )
    bench.add_argument("--repeats", type=int, default=20)
    bench.add_argument("-k", type=int, default=10)
    bench.add_argument(
        "--diversity", type=float, default=0.3, help="MMR strength for the div(ms) column"
    )
    bench.add_argument("--json", action="store_true")
    bench.set_defaults(func=_cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
