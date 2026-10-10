"""Language detection from file extension."""

from __future__ import annotations

from pathlib import Path

EXT_TO_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".rs": "rust",
    ".go": "go",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".sql": "sql",
    ".lua": "lua",
    ".ex": "elixir",
    ".exs": "elixir",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".clj": "clojure",
    ".vue": "vue",
    ".svelte": "svelte",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".md": "markdown",
    ".rst": "rst",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".ini": "ini",
    ".tf": "terraform",
    ".proto": "proto",
}


def language_for(path: str | Path) -> str | None:
    suffix = Path(path).suffix.lower()
    if not suffix:
        name = Path(path).name.lower()
        if name in {"dockerfile"}:
            return "dockerfile"
        if name in {"makefile"}:
            return "makefile"
        return None
    return EXT_TO_LANGUAGE.get(suffix)
