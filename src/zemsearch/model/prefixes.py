"""Exact task-instruction prefixes for EmbeddingGemma 2.

EmbeddingGemma 2 is trained with short task prefixes. For code retrieval
(asymmetric search) the model card specifies:

    query:    ``task: code retrieval | query: {query}``
    document: ``title: {title or filename} | text: {code}``

These strings are load-bearing: getting them wrong silently degrades ranking.
Keep this module dependency-free and unit-tested.
"""

from __future__ import annotations

CODE_RETRIEVAL_TASK = "task: code retrieval"

#: Emitted as the title when a chunk has no natural name.
NO_TITLE = "none"


def format_query(query: str) -> str:
    """Return the exact query-side prefix for code retrieval."""
    return f"{CODE_RETRIEVAL_TASK} | query: {query}"


def format_document(code: str, title: str | None = None) -> str:
    """Return the exact document-side prefix + body for code retrieval.

    ``title`` should be the workspace-relative path (optionally ``path::symbol``).
    A missing/empty title becomes ``none`` per the model card.
    """
    return f"title: {title or NO_TITLE} | text: {code}"
