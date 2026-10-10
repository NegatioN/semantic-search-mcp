"""Shared runtime wiring the model server, store and vector index.

Used by both the CLI and the MCP server so indexing/search behaviour is
identical in either entrypoint.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Self

from .chunking.base import Chunk
from .config import Config, migrate_legacy_store
from .index.codegraph_indexer import CodeGraphIndexer
from .index.indexer import Indexer, IndexReport
from .index.store import SQLiteStore
from .index.vectors import NumpyVectorStore, SearchHit
from .index.watcher import ReindexScheduler
from .model.client import NATIVE_DIM, EmbeddingClient, view
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
    native_dim: int
    storage_dim: int | None
    query_dim: int
    loaded_dim: int | None
    model_repo: str
    server_url: str
    server_managed: bool
    last_reindex: str | None
    by_language: dict[str, int]
    initialized: bool = False
    watching: bool = False
    watch_interval_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "files": self.files,
            "chunks": self.chunks,
            "vectors": self.vectors,
            # native = model output; storage = on-disk index fact; query = view
            # used for search; loaded = dim actually in the in-memory snapshot.
            "native_dim": self.native_dim,
            "storage_dim": self.storage_dim,
            "query_dim": self.query_dim,
            "loaded_dim": self.loaded_dim,
            "dim": self.loaded_dim,  # back-compat alias
            "initialized": self.initialized,
            "model": {
                "repo": self.model_repo,
                "server_url": self.server_url,
                "managed": self.server_managed,
            },
            "last_reindex": self.last_reindex,
            "by_language": self.by_language,
            "reindex": {
                "watching": self.watching,
                "interval_seconds": self.watch_interval_seconds,
            },
        }


class Runtime:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.server_manager: ServerManager | None = None
        self.client: EmbeddingClient | None = None
        self.store: SQLiteStore | None = None
        self.vector = NumpyVectorStore(snippet_chars=config.index.snippet_chars)
        self.watcher: ReindexScheduler | None = None
        self._lock = threading.RLock()

    # -- lifecycle ----------------------------------------------------------
    def open_store(self) -> SQLiteStore:
        if self.store is None:
            migrate_legacy_store(self.config.root, self.config.store_path)
            self.store = SQLiteStore(self.config.store_path)
            self.store.init_schema()
        return self.store

    def start_model(self) -> EmbeddingClient:
        if self.client is not None:
            return self.client
        with self._lock:
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
            )
            if not client.health():
                raise RuntimeError_(f"embeddings server not healthy at {base_url}")
            self.client = client
            return client

    def close(self) -> None:
        self.stop_watching()
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
    def _model_matches(self, store: SQLiteStore) -> bool:
        """True if the stored embeddings were produced by the same model/settings.

        This intentionally ignores the query dimension — that is a view, not a
        property of the stored vectors.
        """
        mc = self.config.model
        stored_repo = store.get_meta("model_repo")
        if stored_repo is None:
            return True  # fresh index
        return stored_repo == mc.repo and (store.get_meta("normalize") == str(mc.normalize).lower())

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
            mc = self.config.model
            if not force and not self._model_matches(store):
                force = True
            stored_dim = store.stored_dim()
            if not force and stored_dim is not None and stored_dim != NATIVE_DIM:
                # Legacy/compact index: upgrade storage to native so all rows agree.
                force = True
            source = self.config.index.source
            stored_source = store.get_meta("index_source")
            if not force and stored_source is not None and stored_source != source:
                force = True  # switching source requires a full rebuild
            if source == "codegraph":
                indexer: Indexer | CodeGraphIndexer = CodeGraphIndexer(
                    self.config.root, self.config.index, client, store
                )
            else:
                indexer = Indexer(self.config.root, self.config.index, client, store)
            report = indexer.index(subpath=subpath, force=force, progress=progress)
            store.set_meta("index_source", source)
            store.set_meta("model_repo", mc.repo)
            store.set_meta("normalize", str(mc.normalize).lower())
            store.set_meta("native_dim", str(NATIVE_DIM))
            store.set_meta("storage_dim", str(NATIVE_DIM))
            store.set_meta("initialized", "true")
            store.set_meta("last_reindex", datetime.now(UTC).isoformat())
            self.reload()
            return report

    def is_initialized(self) -> bool:
        """True once an index has been built (explicitly) for this repo."""
        store = self.open_store()
        if store.get_meta("initialized") == "true":
            return True
        # Backwards-compat: an existing non-empty index counts as initialized.
        return store.stats()["files"] > 0

    def reload(self) -> None:
        """Rebuild the in-memory search snapshot at the configured query dimension.

        Only the leading ``query_dim`` floats of each stored vector are read from
        SQLite and then re-normalized (the MRL view), so the snapshot is
        ``(n, query_dim)`` regardless of native storage. A fresh store is built and
        assigned atomically, so concurrent readers always see a consistent snapshot.
        """
        with self._lock:
            store = self.open_store()
            query_dim = int(self.config.model.query_dim)
            if query_dim < 1 or query_dim > NATIVE_DIM:
                raise RuntimeError_(f"query_dim must be in 1..{NATIVE_DIM}, got {query_dim}")
            stored_dim = store.stored_dim()
            if stored_dim is not None and query_dim > stored_dim:
                raise RuntimeError_(
                    f"query_dim {query_dim} exceeds stored index dimension {stored_dim}; "
                    f"rebuild at native {NATIVE_DIM} with `zemsearch index --force` "
                    "(or the reindex tool with force=true)"
                )
            records = store.load_records(dim=query_dim)
            for record in records:
                record["vector"] = view(record["vector"], query_dim)
            snapshot = NumpyVectorStore(snippet_chars=self.config.index.snippet_chars)
            snapshot.build(records)
            self.vector = snapshot

    # -- watching -----------------------------------------------------------
    def start_watching(self) -> ReindexScheduler:
        """Start debounced + periodic reindexing of the configured roots."""
        if self.watcher is not None and self.watcher.running:
            return self.watcher
        svc = self.config.reindex
        roots = [self.config.resolve_root(r) for r in self.config.index.roots]
        self.watcher = ReindexScheduler(
            roots,
            self._watch_reindex,
            interval_seconds=svc.interval_seconds,
            debounce_seconds=svc.debounce_seconds,
            store_path=self.config.store_path,
        )
        self.watcher.start()
        return self.watcher

    def stop_watching(self) -> None:
        if self.watcher is not None:
            self.watcher.stop()
            self.watcher = None

    def _watch_reindex(self) -> IndexReport | None:
        # Never auto-index a repo that was not explicitly initialized.
        if not self.is_initialized():
            return None
        return self.index()

    # -- search -------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        k: int = 10,
        path: str | None = None,
        min_score: float = 0.0,
        granularity: str = "any",
        diversity: float = 0.0,
    ) -> list[SearchHit]:
        if self.vector.size == 0:
            self.reload()
        # Grab an immutable snapshot so searches are never blocked by reindexing.
        snapshot = self.vector
        client = self.start_model()
        query_vector = view(client.embed_one(format_query(query)), int(self.config.model.query_dim))
        return snapshot.search(
            query_vector,
            k=k,
            min_score=min_score,
            path=path,
            granularity=granularity,
            diversity=diversity,
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
                native_dim=NATIVE_DIM,
                storage_dim=store.stored_dim(),
                query_dim=int(mc.query_dim),
                loaded_dim=self.vector.dim,
                model_repo=store.get_meta("model_repo") or mc.repo,
                server_url=mc.server_url,
                server_managed=mc.manage_server,
                last_reindex=store.get_meta("last_reindex"),
                by_language=stats["by_language"],
                initialized=(store.get_meta("initialized") == "true" or stats["files"] > 0),
                watching=bool(self.watcher and self.watcher.running),
                watch_interval_seconds=float(self.config.reindex.interval_seconds),
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
