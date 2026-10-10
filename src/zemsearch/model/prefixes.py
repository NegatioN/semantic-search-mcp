"""Exact task-instruction prefixes for EmbeddingGemma 2.

EmbeddingGemma 2 is trained with short task prefixes; omitting them still works
but reduces precision. Code search is an **asymmetric** retrieval task, so the
two sides are formatted differently: the task instruction lives on the *query*
only, while documents use the title/text template (the same template web search,
QA and fact-checking use). Getting this wrong degrades ranking silently — it
does not error — so these strings are load-bearing.

Per the model card's "Code search" row (prompt name ``CodeRetrieval``):

    query:    ``task: code retrieval | query: {query}``
    document: ``title: {title or filename} | text: {code}``

Use ``title: none`` when a document has no title.

Sources:
    * https://huggingface.co/google/embeddinggemma-2
      (Best Practices -> 1. Task Instruction Prefixes -> "Code search")
    * https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2

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
