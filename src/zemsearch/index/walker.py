"""Filesystem discovery with .gitignore awareness.

Walks a root, skipping ignored directories, binary files, oversized files and
common build artifacts. ``pathspec`` provides gitwildmatch semantics.
"""

from __future__ import annotations

import os
from pathlib import Path

import pathspec

from ..config import IndexConfig

#: Directories never worth descending into.
PRUNE_DIRS = {
    ".git",
    ".jj",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".zemsearch",
    ".gemma-embedder",
    "dist",
    "build",
    "target",
}

_BINARY_SNIFF_BYTES = 8192


def is_binary(path: Path) -> bool:
    """Return True if ``path`` looks like a binary file (NUL byte in head)."""
    try:
        with path.open("rb") as fh:
            head = fh.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return True
    return b"\x00" in head


def _gitignore_patterns(root: Path) -> list[str]:
    patterns: list[str] = []
    for gitignore in sorted(root.rglob(".gitignore")):
        try:
            rel_dir = gitignore.parent.relative_to(root)
        except ValueError:
            continue
        if any(part in PRUNE_DIRS for part in rel_dir.parts):
            continue
        try:
            lines = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            entry = line.strip()
            if not entry or entry.startswith(("#", "!")):
                continue
            entry = entry.removeprefix("/")
            if rel_dir != Path("."):
                entry = f"{rel_dir.as_posix()}/{entry}"
            patterns.append(entry)
    return patterns


def build_spec(root: Path, cfg: IndexConfig) -> pathspec.PathSpec:
    patterns = list(cfg.exclude)
    if cfg.respect_gitignore:
        patterns.extend(_gitignore_patterns(root))
    return pathspec.PathSpec.from_lines("gitignore", patterns)


def walk(root: Path, cfg: IndexConfig) -> list[Path]:
    """Return a sorted list of indexable files under ``root``."""
    root = root.resolve()
    spec = build_spec(root, cfg)
    results: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        rel_dir = Path(dirpath).relative_to(root)
        dirnames[:] = [d for d in dirnames if not spec.match_file(f"{(rel_dir / d).as_posix()}/")]
        for name in filenames:
            path = Path(dirpath) / name
            rel = (rel_dir / name).as_posix()
            if spec.match_file(rel):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size == 0 or stat.st_size > cfg.max_file_bytes:
                continue
            if is_binary(path):
                continue
            results.append(path)

    return sorted(results)
