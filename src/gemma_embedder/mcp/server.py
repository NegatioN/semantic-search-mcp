"""MCP server exposing semantic code search over the local index.

Uses the official Model Context Protocol Python SDK (v2 ``MCPServer``) over
stdio. All tool output is JSON-serializable dicts.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from .. import __version__
from ..runtime import Runtime

INSTRUCTIONS = (
    "Semantic code search over a locally indexed workspace using "
    "EmbeddingGemma 2. Use `semantic_search` to find relevant code by meaning, "
    "`get_context` to expand a hit into surrounding lines, `reindex` to refresh "
    "the index, and `index_status` to inspect state."
)


def build_server(runtime: Runtime) -> MCPServer:
    server = MCPServer(
        name="gemma-embedder",
        title="gemma-embedder",
        version=__version__,
        instructions=INSTRUCTIONS,
    )

    @server.tool(
        description=(
            "Semantic code search over the indexed workspace. Returns the most "
            "relevant code chunks with file path, line range and a snippet."
        )
    )
    def semantic_search(
        query: str,
        k: int = 10,
        path: str | None = None,
        min_score: float = 0.0,
        granularity: str = "any",
    ) -> dict[str, Any]:
        """Search indexed code by meaning.

        Args:
            query: Natural-language description of what you are looking for.
            k: Maximum number of results (1-50).
            path: Optional workspace-relative path prefix to restrict results.
            min_score: Minimum cosine similarity in [0, 1].
            granularity: One of "any", "file", "symbol".
        """
        k = max(1, min(int(k), 50))
        hits = runtime.search(query, k=k, path=path, min_score=min_score, granularity=granularity)
        return {
            "query": query,
            "count": len(hits),
            "results": [hit.as_dict() for hit in hits],
        }

    @server.tool(description="Return surrounding source lines for a file or a prior search hit.")
    def get_context(
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
        context_lines: int = 20,
    ) -> dict[str, Any]:
        """Read a line range from a workspace file, expanded by context_lines."""
        return runtime.get_context(
            path,
            start_line=int(start_line),
            end_line=int(end_line) if end_line is not None else None,
            context_lines=int(context_lines),
        )

    @server.tool(
        description=(
            "Re-index the workspace (or a subpath). Incremental by default; "
            "set force=true to re-embed everything."
        )
    )
    def reindex(path: str | None = None, force: bool = False) -> dict[str, Any]:
        """Index changed files and refresh the search snapshot."""
        report = runtime.index(subpath=path, force=bool(force))
        return report.as_dict()

    @server.tool(description="Report index and model status.")
    def index_status() -> dict[str, Any]:
        return runtime.status().as_dict()

    return server


def run(runtime: Runtime, transport: str = "stdio") -> None:
    server = build_server(runtime)
    server.run(transport=transport)  # type: ignore[arg-type]
