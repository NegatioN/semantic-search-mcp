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


class TestDiversity(unittest.TestCase):
    """MMR: a near-duplicate should give way to a distinct-but-relevant hit."""

    def setUp(self):
        # query direction is [1,0,0]; two near-identical "dup.py" chunks plus b.
        self.store = NumpyVectorStore()
        self.store.build(
            [
                _rec(1, "dup.py", [0.8, 0.6, 0.0]),
                _rec(2, "dup.py", [0.913811, 0.406138, 0.0]),
                _rec(3, "other.py", [0.707107, -0.707107, 0.0]),
            ]
        )
        self.query = [1.0, 0.0, 0.0]

    def test_diversity_zero_matches_topk(self):
        base = [h.chunk_id for h in self.store.search(self.query, k=2)]
        div0 = [h.chunk_id for h in self.store.search(self.query, k=2, diversity=0.0)]
        self.assertEqual(base, div0)

    def test_diversity_replaces_near_duplicate(self):
        plain = [h.path for h in self.store.search(self.query, k=2)]
        diverse = [h.path for h in self.store.search(self.query, k=2, diversity=0.5)]
        self.assertEqual(plain, ["dup.py", "dup.py"])
        self.assertEqual(diverse, ["dup.py", "other.py"])

    def test_diversity_keeps_top_hit(self):
        top = self.store.search(self.query, k=1)[0].chunk_id
        div_top = self.store.search(self.query, k=1, diversity=0.9)[0].chunk_id
        self.assertEqual(top, div_top)


if __name__ == "__main__":
    unittest.main()
