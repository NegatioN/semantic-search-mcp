"""Shared runtime wiring the model server, store and vector index.

Used by both the CLI and the MCP server so indexing/search behaviour is
identical in either entrypoint.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

from .chunking.base import Chunk
from .config import Config
from .index.indexer import Indexer, IndexReport
from .index.store import SQLiteStore
from .index.vectors import NumpyVectorStore, SearchHit
from .model.client import EmbeddingClient
from .model.prefixes import format_query
from .model.server import INSTALL_HINT, ServerManager, find_binary, parse_host_port


class RuntimeError_(RuntimeError):
    """Runtime configuration/startup error."""


@dataclass
class RuntimeStatus:
    root: str
    files: int
    chunks: int
    vectors: int
    dim: int | None
    model_repo: str
    server_url: str
    server_managed: bool
    last_reindex: str | None
    by_language: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "files": self.files,
            "chunks": self.chunks,
            "vectors": self.vectors,
            "dim": self.dim,
            "model": {
                "repo": self.model_repo,
                "server_url": self.server_url,
                "managed": self.server_managed,
            },
            "last_reindex": self.last_reindex,
            "by_language": self.by_language,
        }


class Runtime:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.server_manager: ServerManager | None = None
        self.client: EmbeddingClient | None = None
        self.store: SQLiteStore | None = None
        self.vector = NumpyVectorStore(snippet_chars=config.index.snippet_chars)
        self._lock = threading.RLock()

    # -- lifecycle ----------------------------------------------------------
    def open_store(self) -> SQLiteStore:
        if self.store is None:
            self.store = SQLiteStore(self.config.store_path)
            self.store.init_schema()
        return self.store

    def start_model(self) -> EmbeddingClient:
        if self.client is not None:
            return self.client
        mc = self.config.model
        base_url = mc.server_url

        if mc.manage_server:
            binary = find_binary(mc.binary or None)
            if not binary:
                raise RuntimeError_(f"no llama binary found. Install with: {INSTALL_HINT}")
            host, _ = parse_host_port(mc.server_url)
            self.server_manager = ServerManager(
                binary,
                model_repo=mc.repo,
                host=host,
                port=0,
                ctx_size=mc.ctx_size,
                gpu_layers=mc.gpu_layers,
                health_timeout=mc.startup_timeout,
            )
            self.server_manager.start()
            base_url = self.server_manager.base_url

        client = EmbeddingClient(
            base_url,
            api_key=mc.api_key or None,
            normalize=mc.normalize,
            dim=mc.dim or None,
        )
        if not client.health():
            raise RuntimeError_(f"embeddings server not healthy at {base_url}")
        self.client = client
        return client

    def close(self) -> None:
        if self.server_manager is not None:
            self.server_manager.stop()
            self.server_manager = None
        if self.store is not None:
            self.store.close()
            self.store = None
        self.client = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- index --------------------------------------------------------------
    def _check_model_meta(self, store: SQLiteStore) -> bool:
        """Return True if the stored index matches the current model settings."""
        mc = self.config.model
        stored_repo = store.get_meta("model_repo")
        stored_dim = store.get_meta("dim")
        stored_norm = store.get_meta("normalize")
        current = {
            "model_repo": mc.repo,
            "dim": str(mc.dim),
            "normalize": str(mc.normalize).lower(),
        }
        if stored_repo is None:
            for key, value in current.items():
                store.set_meta(key, value)
            return True
        compatible = (
            stored_repo == current["model_repo"]
            and stored_dim == current["dim"]
            and stored_norm == current["normalize"]
        )
        return compatible

    def index(
        self,
        *,
        subpath: str | None = None,
        force: bool = False,
        progress: Callable[[str, int], None] | None = None,
    ) -> IndexReport:
        with self._lock:
            store = self.open_store()
            client = self.start_model()
            if not self._check_model_meta(store) and not force:
                force = True
            indexer = Indexer(self.config.root, self.config.index, client, store)
            report = indexer.index(subpath=subpath, force=force, progress=progress)
            mc = self.config.model
            store.set_meta("model_repo", mc.repo)
            store.set_meta("dim", str(mc.dim))
            store.set_meta("normalize", str(mc.normalize).lower())
            import datetime as _dt

            store.set_meta("last_reindex", _dt.datetime.now(_dt.UTC).isoformat())
            self.reload()
            return report

    def reload(self) -> None:
        with self._lock:
            store = self.open_store()
            self.vector.build(store.load_records())

    # -- search -------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        k: int = 10,
        path: str | None = None,
        min_score: float = 0.0,
        granularity: str = "any",
    ) -> list[SearchHit]:
        with self._lock:
            self.open_store()
            if self.vector.size == 0:
                self.reload()
            client = self.start_model()
            query_vector = client.embed_one(format_query(query))
            return self.vector.search(
                query_vector,
                k=k,
                min_score=min_score,
                path=path,
                granularity=granularity,
            )

    # -- introspection ------------------------------------------------------
    def status(self) -> RuntimeStatus:
        with self._lock:
            store = self.open_store()
            stats = store.stats()
            mc = self.config.model
            return RuntimeStatus(
                root=str(self.config.root),
                files=stats["files"],
                chunks=stats["chunks"],
                vectors=self.vector.size,
                dim=self.vector.dim,
                model_repo=store.get_meta("model_repo") or mc.repo,
                server_url=mc.server_url,
                server_managed=mc.manage_server,
                last_reindex=store.get_meta("last_reindex"),
                by_language=stats["by_language"],
            )

    def get_context(
        self,
        path: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
        context_lines: int = 20,
    ) -> dict[str, Any]:
        """Return file content for a line range, expanded by context_lines."""
        target = (self.config.root / path).resolve()
        if self.config.root not in target.parents and target != self.config.root:
            raise RuntimeError_(f"path escapes workspace root: {path}")
        if not target.is_file():
            raise RuntimeError_(f"not a file: {path}")

        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)
        start = max(1, start_line - context_lines)
        end = end_line if end_line is not None else total
        end = min(total, end + context_lines)
        snippet = "\n".join(lines[start - 1 : end])
        return {
            "path": path,
            "start_line": start,
            "end_line": end,
            "total_lines": total,
            "content": snippet,
        }

    def get_chunk(self, chunk_id: int) -> dict[str, Any] | None:
        return self.open_store().get_chunk(chunk_id)


def chunk_titles(chunks: list[Chunk]) -> list[str]:
    return [c.title for c in chunks]
