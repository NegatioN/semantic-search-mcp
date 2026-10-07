"""Console entrypoint for gemma-embedder.

Commands: ``doctor``, ``probe``, ``index``, ``watch``, ``search``, ``status``,
``serve``.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    binary = find_binary(args.binary)
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
    print(f"vectors       : {status.vectors} (dim {status.dim})")
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


def _cmd_serve(args: argparse.Namespace) -> int:
    from .mcp.server import run as run_server

    cfg = _load_config(args)
    with Runtime(cfg) as runtime:
        runtime.open_store()
        runtime.start_model()
        if cfg.reindex.index_on_start:
            report = runtime.index()
            print(
                f"indexed: {report.changed} changed, {report.unchanged} unchanged, "
                f"{report.chunks} chunks",
                file=sys.stderr,
            )
        else:
            runtime.reload()
        if cfg.reindex.watch:
            runtime.start_watching()
            print("watching for changes", file=sys.stderr)
        run_server(runtime, cfg.server.transport)
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="path to gemma-embedder.toml")
    parser.add_argument("--root", default=None, help="workspace root (default: cwd)")
    parser.add_argument("--binary", default=None, help="path to llama/llama-server")
    parser.add_argument("--server-url", default=None)
    parser.add_argument(
        "--no-spawn", action="store_true", help="use an external server, do not spawn"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gemma-embedder")
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
    search.add_argument("--json", action="store_true")
    search.set_defaults(func=_cmd_search)

    status = sub.add_parser("status", help="show index status")
    _add_common(status)
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=_cmd_status)

    serve = sub.add_parser("serve", help="run the MCP server (stdio)")
    _add_common(serve)
    serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
