"""Shared incremental-indexing orchestration for every chunk source.

Both sources plan a set of per-file updates and then run the same mechanical
tail: embed each file's chunks in one batch, replace the file row (chunks +
vectors), delete files whose chunks vanished, and fill in the report.

Only *planning* differs — the file source diffs raw-byte sha256, the codegraph
source diffs enriched-document hashes — so that is the lone abstract step
(:meth:`IncrementalIndexer.plan`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..chunking.base import Chunk
from ..config import IndexConfig
from .store import SQLiteStore

ProgressFn = Callable[[str, int], None]


@dataclass
class FileUpdate:
    """One file to (re)embed: its chunks plus the metadata to persist."""

    path: str
    chunks: list[Chunk]
    mtime: float
    size: int
    sha256: str
    language: str | None = None


@dataclass
class IndexPlan:
    """What a source decided needs writing, before any embedding happens."""

    updates: list[FileUpdate] = field(default_factory=list)
    deletes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    scanned: int = 0
    #: Explicit unchanged count from planning. ``None`` means "derive as
    #: ``scanned - changed`` after applying" (the codegraph source).
    unchanged: int | None = None


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


class IncrementalIndexer:
    """Base: plan updates, then embed + write + delete them."""

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
        plan = self.plan(subpath=subpath, force=force)
        report.errors.extend(plan.errors)
        self._apply(plan, report, progress)
        report.scanned = plan.scanned
        report.unchanged = (
            max(0, plan.scanned - report.changed) if plan.unchanged is None else plan.unchanged
        )
        report.duration_s = time.monotonic() - started
        return report

    def plan(self, *, subpath: str | None, force: bool) -> IndexPlan:
        """Decide which files to update/delete. Implemented by each source."""
        raise NotImplementedError

    def _apply(
        self, plan: IndexPlan, report: IndexReport, progress: ProgressFn | None
    ) -> None:
        for update in plan.updates:
            try:
                vectors = self.client.embed([c.document() for c in update.chunks])
            except Exception as exc:  # noqa: BLE001 - keep indexing resilient
                report.errors.append(f"{update.path}: embed failed: {exc}")
                continue
            self.store.replace_file(
                update.path,
                mtime=update.mtime,
                size=update.size,
                sha256=update.sha256,
                language=update.language,
                chunks=update.chunks,
                vectors=vectors,
            )
            report.changed += 1
            report.chunks += len(update.chunks)
            if progress:
                progress(update.path, len(update.chunks))
        for rel in plan.deletes:
            self.store.delete_file(rel)
            report.deleted += 1
