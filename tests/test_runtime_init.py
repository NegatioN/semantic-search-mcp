import tempfile
import unittest

from gemma_embedder import config as config_mod
from gemma_embedder.chunking.base import Chunk
from gemma_embedder.runtime import Runtime


class TestRuntimeInitialization(unittest.TestCase):
    def test_not_initialized_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            runtime = Runtime(config_mod.load(root=d))
            try:
                self.assertFalse(runtime.is_initialized())
            finally:
                runtime.close()

    def test_explicit_flag_marks_initialized(self):
        with tempfile.TemporaryDirectory() as d:
            runtime = Runtime(config_mod.load(root=d))
            try:
                store = runtime.open_store()
                self.assertFalse(runtime.is_initialized())
                store.set_meta("initialized", "true")
                self.assertTrue(runtime.is_initialized())
            finally:
                runtime.close()

    def test_existing_index_counts_as_initialized(self):
        with tempfile.TemporaryDirectory() as d:
            runtime = Runtime(config_mod.load(root=d))
            try:
                store = runtime.open_store()
                store.replace_file(
                    "a.py",
                    mtime=1.0,
                    size=1,
                    sha256="x",
                    language="python",
                    chunks=[Chunk(path="a.py", title="a.py", text="x = 1")],
                    vectors=[[1.0, 0.0]],
                )
                self.assertTrue(runtime.is_initialized())
            finally:
                runtime.close()


if __name__ == "__main__":
    unittest.main()
