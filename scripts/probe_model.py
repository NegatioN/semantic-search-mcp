#!/usr/bin/env python3
"""Thin wrapper so the Phase 0 spike can run without installing the package."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gemma_embedder.probe import main

if __name__ == "__main__":
    raise SystemExit(main())
