"""Chunk data model."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..model.prefixes import format_document


@dataclass
class Chunk:
    """A unit of code to embed.

    ``text`` is the raw code body; :meth:`document` applies the EmbeddingGemma
    2 document prefix. ``title`` is the path (optionally ``path::symbol``).
    """

    path: str
    title: str
    text: str
    start_line: int = 1
    end_line: int = 1
    granularity: str = "file"
    symbol: str | None = None
    kind: str | None = None
    language: str | None = None

    def document(self) -> str:
        return format_document(self.text, title=self.title)


def document_hash(chunk: Chunk) -> str:
    """Hash of the exact embedded document (prefix included).

    Used for incremental change detection: if two runs produce the same hash
    for a chunk, its embedding is unchanged and does not need to be recomputed.
    """
    return hashlib.sha256(chunk.document().encode("utf-8")).hexdigest()
