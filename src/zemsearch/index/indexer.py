"""Whole-file incremental indexing (the default ``source = "file"``)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..chunking.file import chunk_file
from ..languages import language_for
from .base import FileUpdate, IncrementalIndexer, IndexPlan, IndexReport, ProgressFn
from .walker import walk

__all__ = ["IndexReport", "Indexer", "ProgressFn"]


class Indexer(IncrementalIndexer):
    """Walks roots, chunks changed files (sha256 diff) and updates the store."""

    def plan(self, *, subpath: str | None, force: bool) -> IndexPlan:
        plan = IndexPlan()
        existing = self.store.get_file_hashes()
        seen: set[str] = set()
        unchanged = 0

        for path in self._iter_files(subpath):
            rel = path.relative_to(self.root).as_posix()
            seen.add(rel)
            plan.scanned += 1
            try:
                raw = path.read_bytes()
            except OSError as exc:
                plan.errors.append(f"{rel}: {exc}")
                continue
            digest = hashlib.sha256(raw).hexdigest()
            if not force and existing.get(rel) == digest:
                unchanged += 1
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
            stat = path.stat()
            plan.updates.append(
                FileUpdate(rel, chunks, stat.st_mtime, stat.st_size, digest, language)
            )

        for rel in existing:
            if rel in seen:
                continue
            if subpath and not rel.startswith(Path(subpath).as_posix()):
                continue
            plan.deletes.append(rel)
        plan.unchanged = unchanged
        return plan

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
