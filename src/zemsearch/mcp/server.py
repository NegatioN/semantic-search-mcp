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
    "Semantic code search over a locally indexed workspace using EmbeddingGemma 2. "
    "Use it FIRST for broad or vague questions whose mapping onto the code is "
    "unclear — it surfaces candidate files and, importantly, the codebase's own "
    "vocabulary (internal names that don't appear in the question), which you then "
    "feed into a normal grep/read. One or two `semantic_search` calls plus "
    "`get_context` usually suffice. New repositories are NOT indexed automatically: "
    "if `index_status` reports `initialized` false, ASK THE USER before calling "
    "`reindex`. Fall back to grep/read when you already know a concrete symbol, "
    "file, or exact string."
)


def build_server(runtime: Runtime) -> MCPServer:
    server = MCPServer(
        name="zemsearch",
        title="zemsearch",
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
        diversity: float = 0.0,
    ) -> dict[str, Any]:
        """Search indexed code by meaning.

        Args:
            query: Natural-language description of what you are looking for.
            k: Maximum number of results (1-50).
            path: Optional workspace-relative path prefix to restrict results.
            min_score: Minimum cosine similarity in [0, 1].
            granularity: One of "any", "file", "symbol".
            diversity: MMR strength in [0, 1]; 0 = plain top-k (default), larger
                values trade a little relevance for a wider variety of files.
        """
        k = max(1, min(int(k), 50))
        hits = runtime.search(
            query,
            k=k,
            path=path,
            min_score=min_score,
            granularity=granularity,
            diversity=max(0.0, float(diversity)),
        )
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
            "Build or update the index for this workspace. Call once to "
            "initialize a repo before searching; incremental by default, "
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
