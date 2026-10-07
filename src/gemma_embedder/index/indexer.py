"""Incremental indexing orchestration."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..chunking.file import chunk_file
from ..config import IndexConfig
from ..languages import language_for
from .store import SQLiteStore
from .walker import walk

ProgressFn = Callable[[str, int], None]


@dataclass
class IndexReport:
    scanned: int = 0
    changed: int = 0
    unchanged: int = 0
    deleted: int = 0
    chunks: int = 0
    duration_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "scanned": self.scanned,
            "changed": self.changed,
            "unchanged": self.unchanged,
            "deleted": self.deleted,
            "chunks": self.chunks,
            "duration_s": round(self.duration_s, 2),
            "errors": self.errors,
        }


class Indexer:
    """Walks roots, chunks changed files, embeds them and updates the store."""

    def __init__(self, root: Path, cfg: IndexConfig, client, store: SQLiteStore) -> None:
        self.root = root
        self.cfg = cfg
        self.client = client
        self.store = store

    def index(
        self,
        *,
        subpath: str | None = None,
        force: bool = False,
        progress: ProgressFn | None = None,
    ) -> IndexReport:
        started = time.monotonic()
        report = IndexReport()
        existing = self.store.get_file_hashes()
        seen: set[str] = set()

        for path in self._iter_files(subpath):
            rel = path.relative_to(self.root).as_posix()
            seen.add(rel)
            report.scanned += 1
            try:
                raw = path.read_bytes()
            except OSError as exc:
                report.errors.append(f"{rel}: {exc}")
                continue
            digest = hashlib.sha256(raw).hexdigest()
            if not force and existing.get(rel) == digest:
                report.unchanged += 1
                continue

            language = language_for(rel)
            chunks = chunk_file(
                path,
                rel,
                max_chars=self.cfg.max_chunk_chars,
                overlap_chars=self.cfg.overlap_chars,
                language=language,
            )
            chunks = [c for c in chunks if c.text.strip()]
            if not chunks:
                continue
            try:
                vectors = self.client.embed([c.document() for c in chunks])
            except Exception as exc:  # noqa: BLE001 - keep indexing resilient
                report.errors.append(f"{rel}: embed failed: {exc}")
                continue

            stat = path.stat()
            self.store.replace_file(
                rel,
                mtime=stat.st_mtime,
                size=stat.st_size,
                sha256=digest,
                language=language,
                chunks=chunks,
                vectors=vectors,
            )
            report.changed += 1
            report.chunks += len(chunks)
            if progress:
                progress(rel, len(chunks))

        for rel in existing:
            if rel in seen:
                continue
            if subpath and not rel.startswith(Path(subpath).as_posix()):
                continue
            self.store.delete_file(rel)
            report.deleted += 1

        report.duration_s = time.monotonic() - started
        return report

    def _iter_files(self, subpath: str | None) -> list[Path]:
        roots: list[Path] = []
        if subpath:
            roots.append(self.root / subpath)
        else:
            for entry in self.cfg.roots:
                roots.append(self.root / entry)

        files: list[Path] = []
        for base in roots:
            if not base.exists():
                continue
            files.extend(walk(base, self.cfg))
        return sorted(set(files))
