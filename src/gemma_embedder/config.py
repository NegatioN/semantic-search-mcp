"""Configuration loading for gemma-embedder.

Reads an optional ``gemma-embedder.toml`` from the project root and merges it
over built-in defaults. Uses only the standard library (``tomllib``).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_FILENAME = "gemma-embedder.toml"

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
    ".gemma-embedder/",
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
    repo: str = "ggml-org/embeddinggemma-2-GGUF:Q8_0"
    binary: str = ""
    manage_server: bool = True
    server_url: str = "http://127.0.0.1:8080"
    ctx_size: int = 8192
    gpu_layers: int = 0
    normalize: bool = True
    dim: int = 256
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


@dataclass
class StoreConfig:
    path: str = ".gemma-embedder/index.db"
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

    data: dict = {}
    if config_path.is_file():
        with config_path.open("rb") as fh:
            data = tomllib.load(fh)

    config = Config(root=root_path)
    if "model" in data:
        config.model = _build(ModelConfig, _section(data, "model"))
    if "index" in data:
        config.index = _build(IndexConfig, _section(data, "index"))
    if "store" in data:
        config.store = _build(StoreConfig, _section(data, "store"))
    if "reindex" in data:
        config.reindex = _build(ReindexConfig, _section(data, "reindex"))
    if "server" in data:
        config.server = _build(ServerConfig, _section(data, "server"))

    env_binary = os.environ.get("GEMMA_EMBEDDER_BINARY")
    if env_binary:
        config.model.binary = env_binary
    env_url = os.environ.get("GEMMA_EMBEDDER_SERVER_URL")
    if env_url:
        config.model.server_url = env_url

    return config
