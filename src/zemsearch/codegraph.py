"""Read a CodeGraph SQLite symbol graph and emit embeddable chunks.

The graph (``.codegraph/codegraph.db``) is built by CodeGraph
(https://github.com/colbymchenry/codegraph) — a local, pre-indexed code knowledge
graph — and is treated as a **read-only** source. Every node is embedded as one
``granularity="symbol"`` :class:`Chunk` **except** the structural kinds in
:data:`EXCLUDED_KINDS` (files and imports, which are not code symbols). This is a
*denylist* so new/less-common declaration kinds (``class``, ``trait``, ``module``,
``object``, ``enum``, …) are included across languages automatically rather than
being silently dropped.

Each chunk's text is *graph-enriched*: signature, docstring, 1-hop neighbours
(calls / called-by / creates / references) and the source slice. Folding the
neighbourhood into the embedded text is what lets a vector capture a symbol's
*role*, not just its body — and it means a change to a caller invalidates the
callee's document (see :func:`document_hash`).

This module is dependency-free (stdlib ``sqlite3``) and side-effect free; the
orchestration lives in :mod:`zemsearch.index.codegraph_indexer`.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .chunking.base import Chunk

#: Structural node kinds that are **not** embedded: a file is not a symbol, and
#: an import is just a module path. Everything else becomes an embedding unit.
EXCLUDED_KINDS = frozenset({"file", "import"})

#: SQLite sidecar files (WAL mode) whose mtime/size participate in the
#: change fingerprint. ``-shm`` is deliberately excluded: it is touched on every
#: reader connection and would make the fingerprint flap without any write.
_FINGERPRINT_SIDECARS = ("", "-wal", "-journal")

#: Relation label -> (edge kind, direction). ``in`` means the node is the target.
_RELATIONS: tuple[tuple[str, str, str], ...] = (
    ("calls", "calls", "out"),
    ("called by", "calls", "in"),
    ("creates", "instantiates", "out"),
    ("references", "references", "out"),
    ("referenced by", "references", "in"),
)


def db_fingerprint(db_path: Path) -> tuple:
    """Cheap ``(name, size, mtime_ns)`` fingerprint of the db + WAL sidecars."""
    parts = []
    for suffix in _FINGERPRINT_SIDECARS:
        p = db_path if not suffix else db_path.with_name(db_path.name + suffix)
        try:
            st = p.stat()
            parts.append((p.name, st.st_size, st.st_mtime_ns))
        except OSError:
            parts.append((db_path.name + suffix, -1, -1))
    return tuple(parts)


@dataclass
class CodeGraphReader:
    """Read-only accessor + chunk builder for a codegraph database."""

    db_path: Path
    root: Path
    max_neighbors: int = 12
    max_body_chars: int = 2000
    _source_cache: dict[str, list[str] | None] = field(
        default_factory=dict, init=False, repr=False
    )

    # -- accessibility / readiness -----------------------------------------
    def exists(self) -> bool:
        return self.db_path.is_file()

    def fingerprint(self) -> tuple:
        return db_fingerprint(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def is_ready(self) -> bool:
        """True when CodeGraph reports a coherent snapshot.

        CodeGraph sets ``project_metadata.index_state = 'complete'`` at the end of
        a run; anything else (or a missing table on older schemas) is treated as
        not-ready only when the key exists and is not ``complete``. A database we
        cannot read (writer holding the lock) is also not ready.
        """
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT value FROM project_metadata WHERE key = 'index_state'"
                ).fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            return False
        return row is None or row[0] == "complete"

    def extraction_version(self) -> str:
        """The extractor version that produced the graph (bump => re-embed all)."""
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT value FROM project_metadata "
                    "WHERE key = 'indexed_with_extraction_version'"
                ).fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            return "unknown"
        return row[0] if row else "unknown"

    # -- chunk construction -------------------------------------------------
    def _node_name(self, nodes: dict[str, sqlite3.Row], node_id: str) -> str:
        node = nodes.get(node_id)
        if node is not None:
            return str(node["name"])
        return node_id.split(":", 1)[-1]

    def _edge_names(
        self,
        edges: dict[tuple[str, str], list[tuple[str, str]]],
        nodes: dict[str, sqlite3.Row],
        node_id: str,
        kind: str,
        direction: str,
    ) -> list[str]:
        picked = []
        for edge_kind, other in edges.get((node_id, direction), []):
            if edge_kind != kind or other == node_id:
                continue
            picked.append(self._node_name(nodes, other))
        return sorted(set(picked))[: self.max_neighbors]

    def _source_slice(self, rel: str, start: int, end: int) -> str:
        if rel not in self._source_cache:
            path = self.root / rel
            try:
                self._source_cache[rel] = path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            except OSError:
                self._source_cache[rel] = None
        lines = self._source_cache[rel]
        if not lines:
            return ""
        text = "\n".join(lines[start - 1 : end])
        return text[: self.max_body_chars]

    def _enrich(self, node: sqlite3.Row, relations: dict[str, list[str]]) -> str:
        parts = [f"{node['kind']} {node['qualified_name']}"]
        signature = (node["signature"] or "").strip()
        parts.append(f"signature: {node['name']}{signature}" if signature else f"name: {node['name']}")
        doc = (node["docstring"] or "").strip()
        if doc:
            parts.append(f"doc: {doc}")
        if node["return_type"]:
            parts.append(f"returns: {node['return_type']}")
        parts.append(f"file: {node['file_path']}")
        for label in ("calls", "called by", "creates", "references", "referenced by"):
            values = relations.get(label)
            if values:
                parts.append(f"{label}: " + ", ".join(values))
        body = self._source_slice(
            str(node["file_path"]), int(node["start_line"]), int(node["end_line"])
        )
        if body:
            parts.append(body)
        return "\n".join(parts)

    def chunks(self) -> list[Chunk]:
        """Build one graph-enriched symbol chunk per node (empty if not ready)."""
        if not self.exists():
            return []
        conn = self._connect()
        try:
            nodes = {row["id"]: row for row in conn.execute("SELECT * FROM nodes")}
            edges: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
            for source, target, kind in conn.execute(
                "SELECT source, target, kind FROM edges"
            ):
                edges[(source, "out")].append((kind, target))
                edges[(target, "in")].append((kind, source))
        finally:
            conn.close()

        chunks: list[Chunk] = []
        seen: set[tuple[str, str, str]] = set()
        for node in nodes.values():
            if node["kind"] in EXCLUDED_KINDS:
                continue
            key = (str(node["file_path"]), str(node["qualified_name"]), str(node["kind"]))
            if key in seen:
                continue
            seen.add(key)
            relations = {
                label: self._edge_names(edges, nodes, node["id"], kind, direction)
                for label, kind, direction in _RELATIONS
            }
            chunks.append(
                Chunk(
                    path=str(node["file_path"]),
                    title=f"{node['file_path']}::{node['qualified_name']}",
                    text=self._enrich(node, relations),
                    start_line=int(node["start_line"]),
                    end_line=int(node["end_line"]),
                    granularity="symbol",
                    symbol=str(node["qualified_name"]),
                    kind=str(node["kind"]),
                    language=str(node["language"]) if node["language"] else None,
                )
            )
        return chunks
