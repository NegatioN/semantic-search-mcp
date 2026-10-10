import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from zemsearch import config as config_mod


class TestConfig(unittest.TestCase):
    def test_defaults_without_file(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = config_mod.load(root=d)
            self.assertEqual(cfg.model.query_dim, 256)
            self.assertEqual(cfg.model.repo, "ggml-org/embeddinggemma-2-GGUF:BF16")
            self.assertEqual(cfg.store_path, Path(d) / ".zemsearch/index.db")
            self.assertEqual(cfg.index.source, "file")
            self.assertEqual(cfg.index.codegraph_db, ".codegraph/codegraph.db")

    def test_file_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "zemsearch.toml").write_text(
                '[model]\ndim = 512\nrepo = "custom"\n\n[index]\nmax_chunk_chars = 123\n'
            )
            cfg = config_mod.load(root=d)
            self.assertEqual(cfg.model.query_dim, 512)  # legacy `dim` maps to query_dim
            self.assertEqual(cfg.model.repo, "custom")
            self.assertEqual(cfg.index.max_chunk_chars, 123)
            # untouched defaults preserved
            self.assertTrue(cfg.model.normalize)

    def test_legacy_config_filename_still_read(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "gemma-embedder.toml").write_text('[index]\nmax_chunk_chars = 42\n')
            cfg = config_mod.load(root=d)
            self.assertEqual(cfg.index.max_chunk_chars, 42)

    def test_legacy_env_vars_still_read(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(
                os.environ, {"GEMMA_EMBEDDER_BINARY": "/old/llama-server"}, clear=False
            ):
                cfg = config_mod.load(root=d)
            self.assertEqual(cfg.model.binary, "/old/llama-server")

    def test_migrate_legacy_store(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            legacy = root / ".gemma-embedder"
            legacy.mkdir()
            (legacy / "index.db").write_text("data")
            (legacy / "index.db-wal").write_text("wal")

            moved = config_mod.migrate_legacy_store(root, root / ".zemsearch/index.db")
            self.assertTrue(moved)
            self.assertTrue((root / ".zemsearch/index.db").is_file())
            self.assertTrue((root / ".zemsearch/index.db-wal").is_file())
            self.assertFalse(legacy.exists())

    def test_migrate_legacy_store_noop_when_target_exists(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".gemma-embedder").mkdir()
            (root / ".gemma-embedder/index.db").write_text("old")
            (root / ".zemsearch").mkdir()
            (root / ".zemsearch/index.db").write_text("new")
            self.assertFalse(config_mod.migrate_legacy_store(root, root / ".zemsearch/index.db"))
            self.assertEqual((root / ".zemsearch/index.db").read_text(), "new")

    def test_migrate_ignores_custom_store_path(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".gemma-embedder").mkdir()
            (root / ".gemma-embedder/index.db").write_text("old")
            self.assertFalse(config_mod.migrate_legacy_store(root, root / "custom/db.sqlite"))
            self.assertTrue((root / ".gemma-embedder/index.db").is_file())


if __name__ == "__main__":
    unittest.main()
