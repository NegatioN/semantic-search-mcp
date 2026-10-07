"""Console entrypoint for gemma-embedder.

Phase 0 exposes ``doctor`` (environment check) and ``probe`` (end-to-end
embedding spike). Later phases add ``serve``, ``index``, ``search``, ``status``.
"""

from __future__ import annotations

import argparse
import sys

from . import probe
from .model.client import EmbeddingClient
from .model.server import INSTALL_HINT, find_binary


def _cmd_doctor(args: argparse.Namespace) -> int:
    binary = find_binary(args.binary)
    print(f"llama binary : {binary or 'NOT FOUND'}")
    if not binary:
        print(f"install with : {INSTALL_HINT}")

    healthy = EmbeddingClient(args.server_url).health()
    print(f"server       : {args.server_url}")
    print(f"health       : {'ok' if healthy else 'unreachable'}")
    return 0 if (binary or healthy) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gemma-embedder")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check for a llama binary and running server")
    doctor.add_argument("--binary", default=None)
    doctor.add_argument("--server-url", default="http://127.0.0.1:8080")
    doctor.set_defaults(func=_cmd_doctor)

    probe_parser = sub.add_parser("probe", help="run the Phase 0 end-to-end probe")
    probe.add_arguments(probe_parser)
    probe_parser.set_defaults(func=probe.run_args)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
