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
        # Precomputed for vectorized filtering (avoids Python loops per query).
        self._paths: np.ndarray | None = None
        self._granularities: np.ndarray | None = None

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
            self._paths = None
            self._granularities = None
            return
        self._matrix = np.vstack([np.asarray(r["vector"], dtype=np.float32) for r in records])
        self._dim = int(self._matrix.shape[1])
        self._records = [
            {key: value for key, value in r.items() if key != "vector"} for r in records
        ]
        self._paths = np.array([str(r["path"]) for r in self._records])
        self._granularities = np.array([str(r.get("granularity") or "file") for r in self._records])

    def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int = 10,
        min_score: float = 0.0,
        path: str | None = None,
        granularity: str | None = None,
        diversity: float = 0.0,
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
                mask &= np.char.startswith(self._paths, path)
            if granularity and granularity != "any":
                mask &= self._granularities == granularity
            scores = np.where(mask, scores, -np.inf)
        else:
            mask = None

        k = max(1, min(k, len(self._records)))
        diversity = max(0.0, float(diversity))
        if diversity == 0.0:
            candidates = np.argpartition(-scores, k - 1)[:k]
            order = candidates[np.argsort(-scores[candidates])]
        else:
            order = self._mmr_order(scores, k, diversity, mask)

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

    def _mmr_order(
        self, scores: np.ndarray, k: int, diversity: float, mask: np.ndarray | None
    ) -> np.ndarray:
        """Maximal Marginal Relevance selection order (relevance - diversity·maxsim).

        Bounded to a top-by-relevance candidate pool so cost is independent of the
        corpus size (O(pool · k² · dim)); ``diversity=0`` is handled by the caller.
        """
        valid = np.flatnonzero(mask) if mask is not None else np.arange(len(self._records))
        if valid.size == 0:
            return np.empty(0, dtype=int)

        pool_size = min(valid.size, max(k * 20, 100))
        valid_scores = scores[valid]
        pool_local = np.argpartition(-valid_scores, pool_size - 1)[:pool_size]
        pool_local = pool_local[np.argsort(-valid_scores[pool_local])]
        pool = valid[pool_local]

        selected = [int(pool[0])]
        remaining = pool[1:].tolist()
        selected_vecs = self._matrix[selected].copy()
        while len(selected) < k and remaining:
            rem = np.asarray(remaining, dtype=int)
            sims = self._matrix[rem] @ selected_vecs.T
            combined = scores[rem] - diversity * sims.max(axis=1)
            best = int(np.argmax(combined))
            chosen = remaining.pop(best)
            selected.append(chosen)
            selected_vecs = np.vstack([selected_vecs, self._matrix[chosen]])
        # The diverse *set* is chosen by MMR; present it in relevance order.
        ordered = np.asarray(selected, dtype=int)
        return ordered[np.argsort(-scores[ordered])]
