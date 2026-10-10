import tempfile
import unittest

import numpy as np

from zemsearch import config as config_mod
from zemsearch.chunking.base import Chunk
from zemsearch.index.store import SQLiteStore
from zemsearch.model.client import view
from zemsearch.runtime import Runtime, RuntimeError_


def _unit(vector: np.ndarray) -> list[float]:
    return (vector / np.linalg.norm(vector)).tolist()


def _make_index(root: str, stored_dim: int = 768) -> config_mod.Config:
    cfg = config_mod.load(root=root)
    store = SQLiteStore(cfg.store_path)
    store.init_schema()
    vectors = [
        _unit(np.arange(1, stored_dim + 1, dtype=float)),
        _unit(np.arange(2, stored_dim + 2, dtype=float)),
    ]
    store.replace_file(
        "a.py",
        mtime=1.0,
        size=1,
        sha256="x",
        language="python",
        chunks=[
            Chunk(path="a.py", title="a.py", text="one"),
            Chunk(path="a.py", title="a.py", text="two"),
        ],
        vectors=vectors,
    )
    store.set_meta("model_repo", cfg.model.repo)
    store.set_meta("normalize", "true")
    store.set_meta("storage_dim", str(stored_dim))
    store.set_meta("initialized", "true")
    store.close()
    return cfg


class TestQueryDimView(unittest.TestCase):
    def test_stored_dim_is_index_fact(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _make_index(d)
            store = SQLiteStore(cfg.store_path)
            try:
                self.assertEqual(store.stored_dim(), 768)
            finally:
                store.close()

    def test_prefix_load_matches_full_slice(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _make_index(d)
            store = SQLiteStore(cfg.store_path)
            try:
                full = store.load_records()
                prefix = store.load_records(dim=256)
                self.assertEqual(len(prefix[0]["vector"]), 256)
                self.assertTrue(
                    np.array_equal(prefix[0]["vector"], np.asarray(full[0]["vector"])[:256])
                )
            finally:
                store.close()

    def test_reload_view_is_unit_norm_at_query_dim(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _make_index(d)
            runtime = Runtime(cfg)
            try:
                runtime.reload()
                self.assertEqual(runtime.vector.dim, 256)
                norms = np.linalg.norm(runtime.vector._matrix, axis=1)
                self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))
            finally:
                runtime.close()

    def test_changing_query_dim_needs_no_reindex(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _make_index(d)
            runtime = Runtime(cfg)
            try:
                store = runtime.open_store()
                self.assertTrue(runtime._model_matches(store))
                cfg.model.query_dim = 128
                runtime.reload()
                self.assertEqual(runtime.vector.dim, 128)
                self.assertTrue(runtime._model_matches(store))  # view dim ignored
            finally:
                runtime.close()

    def test_query_dim_above_stored_errors(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _make_index(d, stored_dim=256)
            cfg.model.query_dim = 512
            runtime = Runtime(cfg)
            try:
                with self.assertRaises(RuntimeError_):
                    runtime.reload()
            finally:
                runtime.close()

    def test_load_returns_raw_prefix_view_normalizes(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _make_index(d)
            store = SQLiteStore(cfg.store_path)
            try:
                full = store.load_records()
                prefix = store.load_records(dim=128)
                raw = np.asarray(prefix[0]["vector"])
                self.assertEqual(raw.shape[0], 128)
                # load returns RAW (non-unit) prefix...
                self.assertNotAlmostEqual(float(np.linalg.norm(raw)), 1.0, places=3)
                # ...and the view re-normalizes it, equal to viewing the full vector
                self.assertAlmostEqual(
                    float(np.linalg.norm(view(raw.tolist(), 128))), 1.0, places=5
                )
                self.assertTrue(
                    np.allclose(
                        view(raw.tolist(), 128),
                        view(np.asarray(full[0]["vector"]).tolist(), 128),
                        atol=1e-6,
                    )
                )
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
