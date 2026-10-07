"""SQLite-backed index: file manifest, chunks and embedding vectors.

Vectors are stored as little-endian float32 blobs so they can be loaded into a
single NumPy matrix without per-row conversion.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import numpy as np

from ..chunking.base import Chunk
from ..languages import language_for

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS files (
    id         INTEGER PRIMARY KEY,
    path       TEXT UNIQUE NOT NULL,
    mtime      REAL NOT NULL,
    size       INTEGER NOT NULL,
    sha256     TEXT NOT NULL,
    language   TEXT,
    indexed_at TEXT,
    status     TEXT DEFAULT 'ok'
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    granularity TEXT NOT NULL,
    symbol      TEXT,
    kind        TEXT,
    language    TEXT,
    start_line  INTEGER,
    end_line    INTEGER,
    text_sha256 TEXT,
    content     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    dim      INTEGER NOT NULL,
    vector   BLOB NOT NULL
);
"""


def vector_to_blob(vector: Sequence[float]) -> bytes:
    return np.asarray(vector, dtype="<f4").tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4").copy()


class SQLiteStore:
    """Persistent index store."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Access is serialized by Runtime's lock, but the MCP SDK runs tools in
        # worker threads, so allow the connection to be shared across threads.
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(_SCHEMA)
        self.set_meta("schema_version", SCHEMA_VERSION)
        self.set_meta("last_full_reindex", datetime.now(UTC).isoformat())

    # -- meta ---------------------------------------------------------------
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # -- files --------------------------------------------------------------
    def get_file_hashes(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT path, sha256 FROM files").fetchall()
        return {row["path"]: row["sha256"] for row in rows}

    def get_file_stat(self, rel: str) -> tuple[float, int] | None:
        row = self.conn.execute("SELECT mtime, size FROM files WHERE path = ?", (rel,)).fetchone()
        if row is None:
            return None
        return (row["mtime"], row["size"])

    def replace_file(
        self,
        rel: str,
        *,
        mtime: float,
        size: int,
        sha256: str,
        language: str | None,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """Insert/replace a file and its chunks+vectors atomically."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        now = datetime.now(UTC).isoformat()
        with self.conn:
            self.conn.execute("DELETE FROM files WHERE path = ?", (rel,))
            cur = self.conn.execute(
                "INSERT INTO files(path, mtime, size, sha256, language, indexed_at) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (rel, mtime, size, sha256, language or language_for(rel), now),
            )
            file_id = cur.lastrowid
            for chunk, vector in zip(chunks, vectors):
                ccur = self.conn.execute(
                    "INSERT INTO chunks(file_id, granularity, symbol, kind, language, "
                    "start_line, end_line, content) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        file_id,
                        chunk.granularity,
                        chunk.symbol,
                        chunk.kind,
                        chunk.language or language,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.text,
                    ),
                )
                chunk_id = ccur.lastrowid
                self.conn.execute(
                    "INSERT INTO embeddings(chunk_id, dim, vector) VALUES(?, ?, ?)",
                    (chunk_id, len(vector), vector_to_blob(vector)),
                )
        return len(chunks)

    def delete_file(self, rel: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM files WHERE path = ?", (rel,))

    # -- loading ------------------------------------------------------------
    def stored_dim(self) -> int | None:
        """The dimension the vectors are persisted at (index fact).

        Prefers the explicit ``storage_dim`` meta; falls back to the legacy
        ``dim`` key, then to the per-row ``embeddings.dim`` column.
        """
        meta = self.get_meta("storage_dim") or self.get_meta("dim")
        if meta is not None:
            return int(meta)
        row = self.conn.execute("SELECT dim FROM embeddings LIMIT 1").fetchone()
        return int(row["dim"]) if row else None

    def _assert_single_storage_dim(self) -> None:
        rows = self.conn.execute("SELECT DISTINCT dim FROM embeddings LIMIT 2").fetchall()
        if len(rows) > 1:
            raise RuntimeError(
                "index contains mixed embedding dimensions; run an explicit "
                "reindex (index --force / reindex(force=true)) to rebuild it"
            )

    def load_records(self, dim: int | None = None) -> list[dict[str, Any]]:
        """Load chunks with vectors for the search snapshot.

        If ``dim`` is given, only the leading ``dim`` float32 values of each
        stored vector are read out of SQLite (``substr``), so a smaller query
        dimension never materializes the full native vector. Vectors are returned
        **raw** (not normalized); the caller applies the MRL view.
        """
        self._assert_single_storage_dim()
        if dim is None:
            vector_expr = "e.vector"
            params: tuple = ()
        else:
            vector_expr = "substr(e.vector, 1, ?)"
            params = (dim * 4,)
        rows = self.conn.execute(
            "SELECT c.id AS chunk_id, f.path AS path, c.granularity, c.symbol, "
            "c.kind, c.language, c.start_line, c.end_line, c.content, "
            f"{vector_expr} AS vector "
            "FROM chunks c JOIN files f ON f.id = c.file_id "
            "JOIN embeddings e ON e.chunk_id = c.id",
            params,
        ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            records.append(
                {
                    "chunk_id": row["chunk_id"],
                    "path": row["path"],
                    "granularity": row["granularity"],
                    "symbol": row["symbol"],
                    "kind": row["kind"],
                    "language": row["language"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "content": row["content"],
                    "vector": blob_to_vector(row["vector"]),
                }
            )
        return records

    def get_chunk(self, chunk_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT c.id AS chunk_id, f.path AS path, c.granularity, c.symbol, "
            "c.kind, c.language, c.start_line, c.end_line, c.content "
            "FROM chunks c JOIN files f ON f.id = c.file_id WHERE c.id = ?",
            (chunk_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_file(self, rel: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM files WHERE path = ?", (rel,)).fetchone()
        return dict(row) if row else None

    # -- stats --------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        files = self.conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"]
        chunks = self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        by_lang = {
            row["language"] or "unknown": row["n"]
            for row in self.conn.execute(
                "SELECT language, COUNT(*) AS n FROM files GROUP BY language"
            ).fetchall()
        }
        return {"files": files, "chunks": chunks, "by_language": by_lang}

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def iter_paths(paths: Iterable[Path]) -> list[str]:
    return [p.as_posix() for p in paths]
