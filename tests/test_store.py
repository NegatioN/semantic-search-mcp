import tempfile
import unittest
from pathlib import Path

from gemma_embedder.chunking.base import Chunk
from gemma_embedder.index.store import SQLiteStore


def _chunk(text: str, path: str = "a.py") -> Chunk:
    return Chunk(path=path, title=path, text=text, start_line=1, end_line=1)


class TestSQLiteStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SQLiteStore(Path(self._tmp.name) / "index.db")
        self.store.init_schema()

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_replace_and_load(self):
        n = self.store.replace_file(
            "a.py",
            mtime=1.0,
            size=8,
            sha256="abc",
            language="python",
            chunks=[_chunk("print(1)")],
            vectors=[[1.0, 0.0]],
        )
        self.assertEqual(n, 1)
        records = self.store.load_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["path"], "a.py")
        self.assertEqual(records[0]["vector"].tolist(), [1.0, 0.0])

    def test_replace_drops_old_chunks(self):
        self.store.replace_file(
            "a.py",
            mtime=1.0,
            size=8,
            sha256="abc",
            language="python",
            chunks=[_chunk("old")],
            vectors=[[1.0, 0.0]],
        )
        self.store.replace_file(
            "a.py",
            mtime=2.0,
            size=8,
            sha256="def",
            language="python",
            chunks=[_chunk("new"), _chunk("new2")],
            vectors=[[0.0, 1.0], [1.0, 1.0]],
        )
        self.assertEqual(self.store.stats()["chunks"], 2)
        records = self.store.load_records()
        self.assertEqual([r["content"] for r in records], ["new", "new2"])

    def test_delete_cascades(self):
        self.store.replace_file(
            "a.py",
            mtime=1.0,
            size=8,
            sha256="abc",
            language="python",
            chunks=[_chunk("x")],
            vectors=[[1.0, 0.0]],
        )
        self.store.delete_file("a.py")
        self.assertEqual(self.store.load_records(), [])
        self.assertEqual(self.store.stats()["chunks"], 0)

    def test_hashes_and_meta(self):
        self.store.replace_file(
            "a.py",
            mtime=1.0,
            size=8,
            sha256="abc",
            language="python",
            chunks=[_chunk("x")],
            vectors=[[1.0, 0.0]],
        )
        self.assertEqual(self.store.get_file_hashes(), {"a.py": "abc"})
        self.store.set_meta("model_repo", "test")
        self.assertEqual(self.store.get_meta("model_repo"), "test")


if __name__ == "__main__":
    unittest.main()
