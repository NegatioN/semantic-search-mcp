"""Thin HTTP client for a llama.cpp embeddings server.

Targets the OpenAI-compatible ``POST /v1/embeddings`` endpoint exposed by
``llama-server`` (run with ``--embeddings``). Uses only the standard library so
Phase 0 runs without any third-party dependencies.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Sequence

DEFAULT_MODEL = "embeddinggemma-2"
#: Native EmbeddingGemma 2 output dimension.
NATIVE_DIM = 768


class EmbeddingError(RuntimeError):
    """Raised when the embeddings server returns an error or bad payload."""


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """Return an L2-normalized copy of ``vector`` (no-op for zero vectors)."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return list(vector)
    return [x / norm for x in vector]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product of two equal-length vectors."""
    return sum(x * y for x, y in zip(a, b))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; equivalent to :func:`dot` when inputs are unit-norm."""
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


@dataclass
class WarmupReport:
    """Result of verifying the server actually produces usable embeddings."""

    dim: int
    norm: float
    deterministic: bool

    @property
    def ok(self) -> bool:
        return (
            self.dim == NATIVE_DIM
            and abs(self.norm - 1.0) < 1e-3
            and self.deterministic
        )


class EmbeddingClient:
    """Minimal JSON-over-HTTP client for llama-server embeddings."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        normalize: bool = True,
        dim: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.normalize = normalize
        self.dim = dim
        if dim is not None and dim > NATIVE_DIM:
            raise ValueError(f"dim must be <= {NATIVE_DIM}, got {dim}")

    # -- server lifecycle helpers -------------------------------------------
    def health(self) -> bool:
        """Return True if ``GET /health`` reports the model is loaded."""
        req = urllib.request.Request(f"{self.base_url}/health")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return False
        return body.get("status") == "ok"

    # -- embedding ----------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input in order."""
        if not texts:
            return []
        payload = json.dumps({"model": self.model, "input": list(texts)}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/v1/embeddings", data=payload, headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - server dependent
            detail = exc.read().decode("utf-8", "replace")
            raise EmbeddingError(f"HTTP {exc.code} from embeddings server: {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - server dependent
            raise EmbeddingError(f"cannot reach embeddings server: {exc}") from exc

        data = body.get("data")
        if not isinstance(data, list):
            raise EmbeddingError(f"unexpected embeddings payload: {body!r}")

        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors = [item["embedding"] for item in ordered]
        return [self._postprocess(v) for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _postprocess(self, vector: list[float]) -> list[float]:
        """Apply optional MRL truncation (then re-normalize) and L2 norm."""
        if self.dim is not None:
            vector = vector[: self.dim]
        if self.normalize:
            vector = l2_normalize(vector)
        return vector

    def warmup(self, probe: str = "warmup") -> WarmupReport:
        """Verify dim, unit norm and determinism by embedding a probe twice."""
        first = self.embed_one(probe)
        second = self.embed_one(probe)
        norm = math.sqrt(sum(x * x for x in first))
        return WarmupReport(
            dim=len(first),
            norm=norm,
            deterministic=first == second,
        )
