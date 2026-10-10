"""Configuration loading for zemsearch.

Reads an optional ``zemsearch.toml`` from the project root and merges it
over built-in defaults. Uses only the standard library (``tomllib``).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_FILENAME = "zemsearch.toml"
#: Previous config filename, still read when ``CONFIG_FILENAME`` is absent.
LEGACY_CONFIG_FILENAME = "gemma-embedder.toml"

#: Directory (next to the indexed repo) that holds the embedding index.
STORE_DIRNAME = ".zemsearch"
#: Previous name of the same directory; auto-migrated to :data:`STORE_DIRNAME`.
LEGACY_STORE_DIRNAME = ".gemma-embedder"

#: Patterns always excluded from indexing (in addition to .gitignore).
DEFAULT_EXCLUDES = [
    ".git/",
    "**/.git/**",
    ".jj/",
    "**/.jj/**",
    "node_modules/",
    "**/node_modules/**",
    "__pycache__/",
    "**/__pycache__/**",
    ".venv/",
    "**/.venv/**",
    "venv/",
    "**/venv/**",
    "dist/",
    "**/dist/**",
    "build/",
    "**/build/**",
    "target/",
    "**/target/**",
    f"{STORE_DIRNAME}/",
    f"{LEGACY_STORE_DIRNAME}/",
    ".codegraph/",
    "**/.codegraph/**",
    "*.min.js",
    "*.min.css",
    "*.lock",
    "package-lock.json",
    "uv.lock",
    "poetry.lock",
    "*.gguf",
]


@dataclass
class ModelConfig:
    backend: str = "llama-server"
    repo: str = "ggml-org/embeddinggemma-2-GGUF:BF16"
    binary: str = ""
    manage_server: bool = True
    server_url: str = "http://127.0.0.1:8080"
    ctx_size: int = 8192
    gpu_layers: int = 0
    normalize: bool = True
    #: Search/query dimension (an MRL "view" of the native 768-d vectors). Stored
    #: vectors are always native; changing this never requires a reindex.
    query_dim: int = 256
    api_key: str = ""
    startup_timeout: float = 600.0


@dataclass
class IndexConfig:
    roots: list[str] = field(default_factory=lambda: ["."])
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    respect_gitignore: bool = True
    max_file_bytes: int = 1_000_000
    chunk: str = "file"
    max_chunk_chars: int = 6000
    overlap_chars: int = 600
    snippet_chars: int = 1600
    #: Where chunks come from: "file" (walk + whole-file chunk) or "codegraph"
    #: (read a CodeGraph ``.codegraph/codegraph.db`` symbol graph).
    source: str = "file"
    #: Workspace-relative path to the CodeGraph SQLite database (source above).
    codegraph_db: str = ".codegraph/codegraph.db"
    #: Max 1-hop neighbours per relation folded into a symbol's embedded text.
    codegraph_max_neighbors: int = 12
    #: Cap on the source slice appended to a symbol's enriched text.
    codegraph_max_body_chars: int = 2000


@dataclass
class StoreConfig:
    path: str = f"{STORE_DIRNAME}/index.db"
    backend: str = "numpy"


@dataclass
class ReindexConfig:
    #: At startup, refresh an already-initialized index. A repo that has never
    #: been indexed is left alone until an explicit `reindex`/`index` call.
    update_on_start: bool = True
    watch: bool = True
    interval_seconds: int = 300
    debounce_seconds: float = 2.0


@dataclass
class ServerConfig:
    transport: str = "stdio"


@dataclass
class Config:
    root: Path
    model: ModelConfig = field(default_factory=ModelConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    reindex: ReindexConfig = field(default_factory=ReindexConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @property
    def store_path(self) -> Path:
        p = Path(self.store.path)
        return p if p.is_absolute() else self.root / p

    def resolve_root(self, sub: str | None = None) -> Path:
        base = self.root
        if not sub:
            return base
        p = Path(sub)
        return p if p.is_absolute() else base / p


def migrate_legacy_store(root: Path, store_path: Path) -> bool:
    """Move a legacy ``.zemsearch/`` index to ``.zemsearch/`` (once).

    Only acts on the *default* store location (a custom ``store.path`` is left
    alone). Moves the database plus its SQLite sidecars so an existing index
    keeps working after the directory rename. Returns True if a move happened.
    """
    if store_path.parent != (root / STORE_DIRNAME):
        return False
    legacy_db = root / LEGACY_STORE_DIRNAME / "index.db"
    if store_path.exists() or not legacy_db.exists():
        return False

    store_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm", "-journal"):
        src = Path(str(legacy_db) + suffix)
        if src.exists():
            src.replace(Path(str(store_path) + suffix))
    try:
        (root / LEGACY_STORE_DIRNAME).rmdir()
    except OSError:
        pass
    return True


def _section(data: dict, name: str) -> dict:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def _build(cls, data: dict):
    kwargs = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
    return cls(**kwargs)


def load(path: Path | str | None = None, root: Path | str | None = None) -> Config:
    """Load configuration, searching ``root`` (default: cwd) for the file."""
    root_path = Path(root).resolve() if root else Path.cwd().resolve()

    if path is not None:
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = root_path / config_path
    else:
        config_path = root_path / CONFIG_FILENAME
        if not config_path.is_file():
            legacy = root_path / LEGACY_CONFIG_FILENAME
            if legacy.is_file():
                config_path = legacy

    data: dict = {}
    if config_path.is_file():
        with config_path.open("rb") as fh:
            data = tomllib.load(fh)

    config = Config(root=root_path)
    if "model" in data:
        model_section = _section(data, "model")
        # Back-compat: the old `dim` key now means the query/view dimension.
        if "query_dim" not in model_section and "dim" in model_section:
            model_section = {**model_section, "query_dim": model_section["dim"]}
        config.model = _build(ModelConfig, model_section)
    if "index" in data:
        config.index = _build(IndexConfig, _section(data, "index"))
    if "store" in data:
        config.store = _build(StoreConfig, _section(data, "store"))
    if "reindex" in data:
        config.reindex = _build(ReindexConfig, _section(data, "reindex"))
    if "server" in data:
        config.server = _build(ServerConfig, _section(data, "server"))

    # Legacy env names are still honored so existing setups keep working.
    env_binary = os.environ.get("ZEMSEARCH_BINARY") or os.environ.get("GEMMA_EMBEDDER_BINARY")
    if env_binary:
        config.model.binary = env_binary
    env_url = os.environ.get("ZEMSEARCH_SERVER_URL") or os.environ.get("GEMMA_EMBEDDER_SERVER_URL")
    if env_url:
        config.model.server_url = env_url

    return config
