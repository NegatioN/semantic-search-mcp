"""Whole-file chunking with a line-based sliding window for oversized files."""

from __future__ import annotations

from pathlib import Path

from .base import Chunk


def split_text(
    text: str,
    *,
    max_chars: int = 6000,
    overlap_chars: int = 600,
) -> list[tuple[int, int, str]]:
    """Split ``text`` into ``(start_line, end_line, content)`` windows.

    Windows are line-aligned and limited to roughly ``max_chars`` characters,
    with ``overlap_chars`` of trailing context carried into the next window.
    Small inputs return a single window.
    """
    if len(text) <= max_chars:
        return [(1, text.count("\n") + 1, text)]

    lines = text.splitlines(keepends=True)
    windows: list[tuple[int, int, str]] = []
    start = 0
    total = len(lines)

    while start < total:
        end = start
        size = 0
        while end < total and (size < max_chars or end == start):
            size += len(lines[end])
            end += 1

        windows.append((start + 1, end, "".join(lines[start:end])))
        if end >= total:
            break

        back = end
        overlap = 0
        while back > start and overlap < overlap_chars:
            back -= 1
            overlap += len(lines[back])
        start = max(back, start + 1)

    return windows


def chunk_file(
    path: Path,
    rel: str,
    *,
    max_chars: int = 6000,
    overlap_chars: int = 600,
    language: str | None = None,
) -> list[Chunk]:
    """Read ``path`` and return whole-file chunk(s)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    windows = split_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
    return [
        Chunk(
            path=rel,
            title=rel,
            text=content,
            start_line=start,
            end_line=end,
            granularity="file",
            language=language,
        )
        for start, end, content in windows
    ]
