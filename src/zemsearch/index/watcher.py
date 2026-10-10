"""Filesystem watching and debounced/periodic reindex scheduling.

A watchdog observer pushes events into a queue; a single worker thread drains
them, waits for a quiet period (debounce), then runs one incremental reindex.
A periodic timer also triggers a reindex even without events, which is the
authoritative reconciliation pass for missed events and deletions.

The worker only ever runs one reindex at a time, and the runtime lock serialises
it with on-demand ``reindex`` tool calls.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .walker import PRUNE_DIRS

_IGNORED_SUFFIXES = {".swp", ".swx", ".tmp", ".temp", ".bak"}
_IGNORED_NAMES = {".DS_Store"}
_PROCESSED_EVENT_TYPES = {"created", "modified", "deleted", "moved"}


def should_ignore(path: str, store_path: Path | None = None) -> bool:
    """Return True for paths that must never trigger a reindex."""
    candidate = Path(path)
    if any(part in PRUNE_DIRS for part in candidate.parts):
        return True
    name = candidate.name
    if name in _IGNORED_NAMES:
        return True
    if name.startswith(".#") or (name.startswith("#") and name.endswith("#")):
        return True
    if name.endswith("~") or candidate.suffix in _IGNORED_SUFFIXES:
        return True
    if store_path is not None:
        try:
            if candidate.resolve() == store_path.resolve():
                return True
        except OSError:
            pass
    return False


class _DebounceHandler(FileSystemEventHandler):
    def __init__(self, enqueue: Callable[[], None], ignore: Callable[[str], bool]) -> None:
        self._enqueue = enqueue
        self._ignore = ignore

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory or event.event_type not in _PROCESSED_EVENT_TYPES:
            return
        if self._ignore(event.src_path):
            return
        self._enqueue()


class ReindexScheduler:
    """Debounced + periodic incremental reindex driver."""

    def __init__(
        self,
        roots: list[Path],
        reindex: Callable[[], Any],
        *,
        interval_seconds: float = 0.0,
        debounce_seconds: float = 2.0,
        store_path: Path | None = None,
        on_index: Callable[[Any], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._roots = [Path(r) for r in roots]
        self._reindex = reindex
        self._interval = float(interval_seconds)
        self._debounce = float(debounce_seconds)
        self._store_path = Path(store_path) if store_path else None
        self._on_index = on_index
        self._on_error = on_error

        self._queue: queue.Queue[object] = queue.Queue()
        self._stop = threading.Event()
        self._observer: Observer | None = None
        self._worker: threading.Thread | None = None

        self.reindex_count = 0
        self.last_report: Any | None = None
        self.last_error: Exception | None = None

    @property
    def running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()

        observer = Observer()
        handler = _DebounceHandler(self.trigger, self._ignore)
        scheduled = 0
        for root in self._roots:
            if root.exists():
                observer.schedule(handler, str(root), recursive=True)
                scheduled += 1
        self._observer = observer if scheduled else None

        self._worker = threading.Thread(target=self._loop, name="zemsearch-reindex", daemon=True)
        self._worker.start()
        if self._observer is not None:
            self._observer.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._queue.put(None)
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout)
            self._observer = None
        if self._worker is not None:
            self._worker.join(timeout)
            self._worker = None

    def trigger(self) -> None:
        """Request a debounced reindex."""
        self._queue.put(None)

    def _ignore(self, path: str) -> bool:
        return should_ignore(path, self._store_path)

    def _loop(self) -> None:
        last_run = time.monotonic()
        while not self._stop.is_set():
            try:
                self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._interval > 0 and (time.monotonic() - last_run) >= self._interval:
                    self._run()
                    last_run = time.monotonic()
                continue

            deadline = time.monotonic() + self._debounce
            while not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    self._queue.get(timeout=remaining)
                    deadline = time.monotonic() + self._debounce
                except queue.Empty:
                    break

            if self._stop.is_set():
                break
            self._run()
            last_run = time.monotonic()

    def _run(self) -> None:
        try:
            report = self._reindex()
            self.reindex_count += 1
            self.last_report = report
            if self._on_index is not None:
                self._on_index(report)
        except Exception as exc:  # noqa: BLE001 - keep the watcher alive
            self.last_error = exc
            if self._on_error is not None:
                self._on_error(exc)
