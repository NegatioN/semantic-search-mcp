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

The mechanical embed/write/delete tail is shared via :class:`IncrementalIndexer`;
only :meth:`CodeGraphIndexer.plan` (the hash diff and its readiness gates) is
specific to this source.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from ..chunking.base import Chunk, document_hash
from ..codegraph import CodeGraphReader
from ..config import IndexConfig
from .base import FileUpdate, IncrementalIndexer, IndexPlan
from .store import SQLiteStore


class CodeGraphIndexer(IncrementalIndexer):
    """Mirror a codegraph SQLite graph into the embedding store (incremental)."""

    def __init__(self, root: Path, cfg: IndexConfig, client, store: SQLiteStore) -> None:
        super().__init__(root, cfg, client, store)
        self.reader = CodeGraphReader(
            root / cfg.codegraph_db,
            root,
            max_neighbors=cfg.codegraph_max_neighbors,
            max_body_chars=cfg.codegraph_max_body_chars,
        )

    def plan(self, *, subpath: str | None, force: bool) -> IndexPlan:
        plan = IndexPlan()
        if not self.reader.exists():
            plan.errors.append(f"codegraph db not found: {self.reader.db_path}")
            return plan
        if not self.reader.is_ready():
            # CodeGraph is mid-write; retry on the next scheduler pass.
            return plan

        extraction = self.reader.extraction_version()
        stored_extraction = self.store.get_meta("codegraph_extraction_version")
        if stored_extraction is not None and stored_extraction != extraction:
            force = True  # enrichment semantics changed; re-embed everything

        fingerprint = repr(self.reader.fingerprint())
        stored_fp = self.store.get_meta("codegraph_fingerprint")
        if not force and stored_fp == fingerprint and self.store.get_symbol_hashes():
            return plan

        chunks = self.reader.chunks()
        if subpath:
            prefix = Path(subpath).as_posix()
            chunks = [
                c for c in chunks if c.path == prefix or c.path.startswith(prefix + "/")
            ]

        by_file: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            by_file[chunk.path].append(chunk)
        plan.scanned = len(by_file)

        new_hashes = {(c.path, c.symbol or "", c.kind or ""): document_hash(c) for c in chunks}
        old_hashes = {} if force else self.store.get_symbol_hashes()

        dirty: set[str] = set()
        for key, digest in new_hashes.items():
            if old_hashes.get(key) != digest:
                dirty.add(key[0])
        for key in old_hashes:
            if key not in new_hashes:
                dirty.add(key[0])  # a symbol disappeared -> its file needs rewriting

        for rel in sorted(dirty):
            file_chunks = by_file.get(rel, [])
            if not file_chunks:
                continue
            mtime, size = self._file_stat(rel, file_chunks)
            plan.updates.append(
                FileUpdate(
                    rel,
                    file_chunks,
                    mtime,
                    size,
                    self._file_digest(file_chunks),
                    file_chunks[0].language,
                )
            )
        for rel in dirty:
            if rel not in by_file:
                plan.deletes.append(rel)

        self.store.set_meta("codegraph_fingerprint", fingerprint)
        self.store.set_meta("codegraph_extraction_version", extraction)
        return plan

    def _file_stat(self, rel: str, chunks: list[Chunk]) -> tuple[float, int]:
        path = self.root / rel
        try:
            st = path.stat()
            return st.st_mtime, st.st_size
        except OSError:
            return 0.0, sum(len(c.text) for c in chunks)

    def _file_digest(self, chunks: list[Chunk]) -> str:
        return hashlib.sha256(
            "\n".join(sorted(document_hash(c) for c in chunks)).encode("utf-8")
        ).hexdigest()
