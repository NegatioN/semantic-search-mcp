"""Filesystem discovery with a single gitignore-aware scope.

One gitignore-semantics matcher (``pathspec.GitIgnoreSpec``) is the sole source of
truth for what to skip. It is built from, in order:

    built-in defaults (``cfg.exclude``)
    + every ``.gitignore`` under the root (nested patterns are re-scoped, and
      negations are respected)
    + git's own root-relative exclude files (``.git/info/exclude`` and
      ``core.excludesFile``)

The walker and the watcher share the same :class:`FileScope`, so both prune
identically. Directory pruning is a property of the matcher (a ``dir/`` pattern
matches at any depth) — there is no separate hard-coded name list to drift.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pathspec

from ..config import IndexConfig

_BINARY_SNIFF_BYTES = 8192
_GIT_CONFIG_TIMEOUT = 5.0


def is_binary(path: Path) -> bool:
    """Return True if ``path`` looks like a binary file (NUL byte in head)."""
    try:
        with path.open("rb") as fh:
            head = fh.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return True
    return b"\x00" in head


def make_spec(patterns: list[str]) -> pathspec.GitIgnoreSpec:
    """Build a gitignore spec, dropping malformed lines instead of raising.

    A stray ``.gitignore`` line (e.g. a lone ``!``) would otherwise abort the
    whole walk; mirror git/CodeGraph by skipping just the bad pattern.
    """
    try:
        return pathspec.GitIgnoreSpec.from_lines("gitignore", patterns)
    except ValueError:
        good: list[str] = []
        for pattern in patterns:
            try:
                pathspec.GitIgnoreSpec.from_lines("gitignore", [pattern])
            except ValueError:
                continue
            good.append(pattern)
        return pathspec.GitIgnoreSpec.from_lines("gitignore", good)


def _read_ignore_lines(path: Path) -> list[str]:
    """Read a gitignore file's effective lines, never raising."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines: list[str] = []
    for raw in text.splitlines():
        entry = raw.strip()
        if not entry or entry.startswith("#") or entry == "!":
            continue
        lines.append(entry)
    return lines


def _scope_nested(rel_dir: str, pattern: str) -> str:
    """Re-scope a nested ``.gitignore`` pattern to the walk root.

    Git treats a pattern with a slash (or a leading one) as anchored to the
    ignore file's directory, and a bare pattern as matching at any depth below
    it. Reproduce that when flattening into a single root-relative spec.
    """
    negated = pattern.startswith("!")
    body = pattern[1:] if negated else pattern
    anchored = body.startswith("/")
    if anchored:
        body = body[1:]
    trailing = body.endswith("/")
    core = body[:-1] if trailing else body
    if not core:
        return pattern
    if anchored or "/" in core:
        scoped = f"{rel_dir}/{core}"
    else:
        scoped = f"{rel_dir}/**/{core}"
    if trailing:
        scoped += "/"
    return ("!" if negated else "") + scoped


def _nested_gitignore_patterns(root: Path, defaults: pathspec.PathSpec) -> list[str]:
    """Collect ``.gitignore`` lines under ``root``, pruning default-ignored dirs."""
    patterns: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        dirnames[:] = [
            d for d in dirnames if not defaults.match_file(f"{(rel / d).as_posix()}/")
        ]
        if ".gitignore" not in filenames:
            continue
        gi = Path(dirpath) / ".gitignore"
        if rel == Path("."):
            patterns.extend(_read_ignore_lines(gi))
        else:
            patterns.extend(_scope_nested(rel.as_posix(), p) for p in _read_ignore_lines(gi))
    return patterns


def _resolve_git_dir(root: Path) -> Path | None:
    """Return ``root``'s git dir, following a ``gitdir:`` pointer for worktrees."""
    dot_git = root / ".git"
    if dot_git.is_dir():
        return dot_git
    if dot_git.is_file():
        try:
            first = dot_git.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return None
        if first.startswith("gitdir:"):
            target = Path(first.split(":", 1)[1].strip())
            if not target.is_absolute():
                target = (root / target).resolve()
            return target if target.is_dir() else None
    return None


def _git_extra_excludes(root: Path) -> list[str]:
    """Patterns from ``.git/info/exclude`` and git's ``core.excludesFile``."""
    patterns: list[str] = []
    git_dir = _resolve_git_dir(root)
    if git_dir is not None:
        info_exclude = git_dir / "info" / "exclude"
        if info_exclude.is_file():
            patterns.extend(_read_ignore_lines(info_exclude))

    git = shutil.which("git")
    if git:
        try:
            result = subprocess.run(
                [git, "-C", str(root), "config", "--get", "core.excludesFile"],
                capture_output=True,
                text=True,
                timeout=_GIT_CONFIG_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0 and result.stdout.strip():
            excludes = Path(os.path.expanduser(result.stdout.strip()))
            if not excludes.is_absolute():
                excludes = root / excludes
            if excludes.is_file():
                patterns.extend(_read_ignore_lines(excludes))
    return patterns


@dataclass
class FileScope:
    """A root plus the gitignore-semantics matcher defining indexable paths."""

    root: Path
    spec: pathspec.GitIgnoreSpec

    def rel(self, path: str | Path) -> str | None:
        """Workspace-relative posix path of ``path``, or None if outside root."""
        try:
            return Path(path).resolve().relative_to(self.root).as_posix()
        except (ValueError, OSError):
            return None

    def ignores(self, rel_path: str) -> bool:
        return bool(self.spec.match_file(rel_path))

    def ignores_dir(self, rel_dir: str) -> bool:
        return bool(self.spec.match_file(rel_dir.rstrip("/") + "/"))


def build_scope(root: Path, cfg: IndexConfig) -> FileScope:
    """Build the :class:`FileScope` for ``root`` from defaults + git ignore files."""
    root = root.resolve()
    patterns = list(cfg.exclude)
    if cfg.respect_gitignore:
        defaults = make_spec(cfg.exclude)
        patterns.extend(_nested_gitignore_patterns(root, defaults))
        patterns.extend(_git_extra_excludes(root))
    return FileScope(root=root, spec=make_spec(patterns))


def walk(root: Path, cfg: IndexConfig) -> list[Path]:
    """Return a sorted list of indexable files under ``root``."""
    scope = build_scope(root, cfg)
    root = scope.root
    results: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        dirnames[:] = [
            d for d in dirnames if not scope.ignores_dir((rel_dir / d).as_posix())
        ]
        for name in filenames:
            rel = (rel_dir / name).as_posix()
            if scope.ignores(rel):
                continue
            path = Path(dirpath) / name
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
