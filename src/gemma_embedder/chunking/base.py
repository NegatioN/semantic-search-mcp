"""Chunk data model."""

from __future__ import annotations

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
