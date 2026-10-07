"""Locate, spawn and health-check a local llama.cpp embeddings server.

Only the standard library is used. The manager never downloads or runs remote
scripts on its own; if no binary is found it reports the exact install command
for the user to run.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Self

#: OpenAI-compatible endpoint the server should expose.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_MODEL_REPO = "ggml-org/embeddinggemma-2-GGUF:Q8_0"

INSTALL_HINT = "curl -LsSf https://llama.app/install.sh | sh"


def find_binary(preferred: str | None = None) -> str | None:
    """Return a path to a usable llama binary, or None.

    Accepts either the unified ``llama`` binary or the standalone
    ``llama-server``.
    """
    if preferred:
        if os.path.isfile(preferred) and os.access(preferred, os.X_OK):
            return preferred
        found = shutil.which(preferred)
        if found:
            return found
        return None
    for name in ("llama", "llama-server"):
        found = shutil.which(name)
        if found:
            return found
    return None


def find_free_port() -> int:
    """Return an available TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((DEFAULT_HOST, 0))
        return sock.getsockname()[1]


def parse_host_port(url: str) -> tuple[str, int]:
    """Split a base URL into ``(host, port)`` with sane defaults."""
    parsed = urllib.parse.urlparse(url)
    return parsed.hostname or DEFAULT_HOST, parsed.port or DEFAULT_PORT


def build_command(
    binary: str,
    *,
    model_repo: str = DEFAULT_MODEL_REPO,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    ctx_size: int = 8192,
    batch_size: int | None = None,
    ubatch_size: int | None = None,
    gpu_layers: int = 0,
    api_key: str | None = None,
) -> list[str]:
    """Build the server argv for either ``llama serve`` or ``llama-server``."""
    cmd = [binary]
    if os.path.basename(binary) == "llama":
        cmd.append("serve")
    cmd += [
        "-hf",
        model_repo,
        "--embeddings",
        "--pooling",
        "mean",
        "--embd-normalize",
        "2",
        "--ctx-size",
        str(ctx_size),
        "--batch-size",
        str(batch_size or ctx_size),
        "--ubatch-size",
        str(ubatch_size or ctx_size),
        "--host",
        host,
        "--port",
        str(port),
    ]
    if gpu_layers:
        cmd += ["--n-gpu-layers", str(gpu_layers)]
    if api_key:
        cmd += ["--api-key", api_key]
    return cmd


class ServerManager:
    """Context manager that spawns and tears down a llama.cpp server.

    If ``binary`` is None the manager only points at ``server_url`` and never
    spawns anything (use this against an already-running server).
    """

    def __init__(
        self,
        binary: str | None = None,
        *,
        model_repo: str = DEFAULT_MODEL_REPO,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        ctx_size: int = 8192,
        gpu_layers: int = 0,
        health_timeout: float = 600.0,
        poll_interval: float = 1.0,
    ) -> None:
        self.binary = binary
        self.model_repo = model_repo
        self.host = host
        self.port = port
        self.ctx_size = ctx_size
        self.gpu_layers = gpu_layers
        self.health_timeout = health_timeout
        self.poll_interval = poll_interval
        self.process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self.binary is None:
            return
        if self.port == 0:
            self.port = find_free_port()
        cmd = build_command(
            self.binary,
            model_repo=self.model_repo,
            host=self.host,
            port=self.port,
            ctx_size=self.ctx_size,
            gpu_layers=self.gpu_layers,
        )
        self.process = subprocess.Popen(
            cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
        )
        if not self.wait_healthy():
            exit_code = self.process.poll()
            self.stop()
            detail = (
                f"process exited with code {exit_code}"
                if exit_code is not None
                else f"no healthy response within {self.health_timeout}s"
            )
            raise RuntimeError(f"embeddings server failed to start: {detail}")

    def wait_healthy(self) -> bool:
        deadline = time.monotonic() + self.health_timeout
        url = f"{self.base_url}/health"
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                if resp.status == 200 and body.get("status") in ("ok", True):
                    return True
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(self.poll_interval)
        return False

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.send_signal(signal.SIGINT)
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover
            self.process.kill()
            self.process.wait(timeout=15)
        finally:
            self.process = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()
