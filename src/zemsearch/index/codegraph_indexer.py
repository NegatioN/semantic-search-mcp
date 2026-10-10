"""Incremental indexing of a CodeGraph symbol graph.

CodeGraph (https://github.com/colbymchenry/codegraph) owns the graph; we only
mirror it into the embedding index. The sync is a two-level diff so the
(expensive) embedder is called only for symbols whose *enriched document*
actually changed:

1. build the current symbol set from the graph and hash each embedded document;
2. compare against the stored hashes keyed by ``(path, symbol, kind)``;
3. re-embed/replace the files that own a changed symbol, delete files whose
   symbols vanished, leave everything else untouched.

Because the embedded text folds in 1-hop neighbours, editing a caller dirties the
callee too — the enriched-document hash catches that even though the callee's own
file did not change. Callers must supply a ready graph
(:meth:`CodeGraphReader.is_ready`) — the scheduler retries otherwise.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from ..chunking.base import Chunk, document_hash
from ..codegraph import CodeGraphReader
from ..config import IndexConfig
from .indexer import IndexReport
from .store import SQLiteStore

ProgressFn = Callable[[str, int], None]


class CodeGraphIndexer:
    """Mirror a codegraph SQLite graph into the embedding store (incremental)."""

    def __init__(
        self, root: Path, cfg: IndexConfig, client, store: SQLiteStore
    ) -> None:
        self.root = root
        self.cfg = cfg
        self.client = client
        self.store = store
        self.reader = CodeGraphReader(
            root / cfg.codegraph_db,
            root,
            max_neighbors=cfg.codegraph_max_neighbors,
            max_body_chars=cfg.codegraph_max_body_chars,
        )

    def index(
        self,
        *,
        subpath: str | None = None,
        force: bool = False,
        progress: ProgressFn | None = None,
    ) -> IndexReport:
        started = time.monotonic()
        report = IndexReport()

        if not self.reader.exists():
            report.errors.append(f"codegraph db not found: {self.reader.db_path}")
            report.duration_s = time.monotonic() - started
            return report
        if not self.reader.is_ready():
            # CodeGraph is mid-write; retry on the next scheduler pass.
            report.duration_s = time.monotonic() - started
            return report

        extraction = self.reader.extraction_version()
        stored_extraction = self.store.get_meta("codegraph_extraction_version")
        if stored_extraction is not None and stored_extraction != extraction:
            force = True  # enrichment semantics changed; re-embed everything

        fingerprint = repr(self.reader.fingerprint())
        stored_fp = self.store.get_meta("codegraph_fingerprint")
        if not force and stored_fp == fingerprint and self.store.get_symbol_hashes():
            report.duration_s = time.monotonic() - started
            return report

        chunks = self.reader.chunks()
        if subpath:
            prefix = Path(subpath).as_posix()
            chunks = [
                c for c in chunks if c.path == prefix or c.path.startswith(prefix + "/")
            ]

        by_file: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            by_file[chunk.path].append(chunk)
        report.scanned = len(by_file)

        new_hashes = {(c.path, c.symbol or "", c.kind or ""): document_hash(c) for c in chunks}
        old_hashes = {} if force else self.store.get_symbol_hashes()

        dirty: set[str] = set()
        for key, digest in new_hashes.items():
            if old_hashes.get(key) != digest:
                dirty.add(key[0])
        for key in old_hashes:
            if key not in new_hashes:
                dirty.add(key[0])  # a symbol disappeared -> its file needs rewriting

        pending: list[tuple[str, list[Chunk], list[list[float]]]] = []
        for rel in sorted(dirty):
            file_chunks = by_file.get(rel, [])
            if not file_chunks:
                continue
            try:
                vectors = self.client.embed([c.document() for c in file_chunks])
            except Exception as exc:  # noqa: BLE001 - keep indexing resilient
                report.errors.append(f"{rel}: embed failed: {exc}")
                continue
            pending.append((rel, file_chunks, vectors))

        for rel, file_chunks, vectors in pending:
            self._write_file(rel, file_chunks, vectors)
            report.changed += 1
            report.chunks += len(file_chunks)
            if progress:
                progress(rel, len(file_chunks))

        for rel in dirty:
            if rel not in by_file:
                self.store.delete_file(rel)
                report.deleted += 1

        report.unchanged = max(0, report.scanned - report.changed)
        self.store.set_meta("codegraph_fingerprint", fingerprint)
        self.store.set_meta("codegraph_extraction_version", extraction)
        report.duration_s = time.monotonic() - started
        return report

    def _write_file(self, rel: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        path = self.root / rel
        try:
            st = path.stat()
            mtime, size = st.st_mtime, st.st_size
        except OSError:
            mtime, size = 0.0, sum(len(c.text) for c in chunks)
        digest = hashlib.sha256(
            "\n".join(sorted(document_hash(c) for c in chunks)).encode("utf-8")
        ).hexdigest()
        self.store.replace_file(
            rel,
            mtime=mtime,
            size=size,
            sha256=digest,
            language=chunks[0].language,
            chunks=chunks,
            vectors=vectors,
        )
