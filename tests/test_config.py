import tempfile
import unittest
from pathlib import Path

from gemma_embedder import config as config_mod


class TestConfig(unittest.TestCase):
    def test_defaults_without_file(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = config_mod.load(root=d)
            self.assertEqual(cfg.model.query_dim, 256)
            self.assertEqual(cfg.model.repo, "ggml-org/embeddinggemma-2-GGUF:BF16")
            self.assertEqual(cfg.store_path, Path(d) / ".gemma-embedder/index.db")

    def test_file_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "gemma-embedder.toml").write_text(
                '[model]\ndim = 512\nrepo = "custom"\n\n[index]\nmax_chunk_chars = 123\n'
            )
            cfg = config_mod.load(root=d)
            self.assertEqual(cfg.model.query_dim, 512)  # legacy `dim` maps to query_dim
            self.assertEqual(cfg.model.repo, "custom")
            self.assertEqual(cfg.index.max_chunk_chars, 123)
            # untouched defaults preserved
            self.assertTrue(cfg.model.normalize)


if __name__ == "__main__":
    unittest.main()
