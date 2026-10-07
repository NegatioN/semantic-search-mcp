"""In-memory NumPy vector store and search.

All vectors are L2-normalized, so a plain dot product ranks identically to
cosine similarity. Exact brute force is used: fine for local repositories.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SearchHit:
    chunk_id: int
    path: str
    title: str
    symbol: str | None
    kind: str | None
    language: str | None
    start_line: int
    end_line: int
    score: float
    snippet: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "symbol": self.symbol,
            "kind": self.kind,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "score": round(self.score, 4),
            "snippet": self.snippet,
        }


class NumpyVectorStore:
    """Immutable snapshot of vectors + metadata, rebuilt on reindex."""

    def __init__(self, snippet_chars: int = 1600) -> None:
        self.snippet_chars = snippet_chars
        self._matrix: np.ndarray | None = None
        self._records: list[dict[str, Any]] = []
        self._dim: int | None = None

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def dim(self) -> int | None:
        return self._dim

    def build(self, records: Sequence[dict[str, Any]]) -> None:
        """Replace the snapshot from ``records`` (each with a ``vector`` key)."""
        if not records:
            self._matrix = None
            self._records = []
            self._dim = None
            return
        self._matrix = np.vstack([np.asarray(r["vector"], dtype=np.float32) for r in records])
        self._dim = int(self._matrix.shape[1])
        self._records = [
            {key: value for key, value in r.items() if key != "vector"} for r in records
        ]

    def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int = 10,
        min_score: float = 0.0,
        path: str | None = None,
        granularity: str | None = None,
    ) -> list[SearchHit]:
        if self._matrix is None or not self._records:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        if self._dim is not None and query.shape[0] != self._dim:
            raise ValueError(f"query dim {query.shape[0]} != index dim {self._dim}")

        scores = self._matrix @ query

        if path or (granularity and granularity != "any"):
            mask = np.ones(len(self._records), dtype=bool)
            if path:
                mask &= np.array([str(r["path"]).startswith(path) for r in self._records])
            if granularity and granularity != "any":
                mask &= np.array([r["granularity"] == granularity for r in self._records])
            scores = np.where(mask, scores, -np.inf)
        else:
            mask = None

        k = max(1, min(k, len(self._records)))
        candidates = np.argpartition(-scores, k - 1)[:k]
        order = candidates[np.argsort(-scores[candidates])]

        hits: list[SearchHit] = []
        for i in order:
            score = float(scores[int(i)])
            if score < min_score:
                continue
            if mask is not None and not mask[int(i)]:
                continue
            record = self._records[int(i)]
            content = str(record.get("content", ""))
            hits.append(
                SearchHit(
                    chunk_id=int(record["chunk_id"]),
                    path=str(record["path"]),
                    title=str(record.get("symbol") or record["path"]),
                    symbol=record.get("symbol"),
                    kind=record.get("kind"),
                    language=record.get("language"),
                    start_line=int(record.get("start_line") or 1),
                    end_line=int(record.get("end_line") or 1),
                    score=score,
                    snippet=content[: self.snippet_chars],
                )
            )
        return hits
