import unittest

from gemma_embedder.index.vectors import NumpyVectorStore


def _rec(chunk_id, path, vector, granularity="file", symbol=None):
    return {
        "chunk_id": chunk_id,
        "path": path,
        "granularity": granularity,
        "symbol": symbol,
        "kind": None,
        "language": None,
        "start_line": 1,
        "end_line": 1,
        "content": f"content {chunk_id}",
        "vector": vector,
    }


class TestNumpyVectorStore(unittest.TestCase):
    def setUp(self):
        self.store = NumpyVectorStore()
        self.store.build(
            [
                _rec(1, "a.py", [1.0, 0.0]),
                _rec(2, "b.py", [0.0, 1.0]),
                _rec(3, "sub/c.py", [0.7071, 0.7071], granularity="symbol", symbol="f"),
            ]
        )

    def test_ranking(self):
        hits = self.store.search([1.0, 0.0], k=3)
        self.assertEqual(hits[0].path, "a.py")
        self.assertAlmostEqual(hits[0].score, 1.0, places=5)
        self.assertEqual(hits[1].path, "sub/c.py")

    def test_path_filter(self):
        hits = self.store.search([1.0, 0.0], k=3, path="sub/")
        self.assertEqual([h.path for h in hits], ["sub/c.py"])

    def test_granularity_filter(self):
        hits = self.store.search([1.0, 0.0], k=3, granularity="symbol")
        self.assertEqual([h.path for h in hits], ["sub/c.py"])

    def test_min_score(self):
        hits = self.store.search([1.0, 0.0], k=3, min_score=0.9)
        self.assertEqual([h.path for h in hits], ["a.py"])

    def test_dim_mismatch(self):
        with self.assertRaises(ValueError):
            self.store.search([1.0, 0.0, 0.0], k=1)

    def test_empty_store(self):
        empty = NumpyVectorStore()
        empty.build([])
        self.assertEqual(empty.search([1.0, 0.0], k=3), [])


if __name__ == "__main__":
    unittest.main()
