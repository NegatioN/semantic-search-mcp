import unittest

from gemma_embedder.bench import format_table, run_bench


class TestBench(unittest.TestCase):
    def test_run_bench_small(self):
        rows = run_bench([100, 1000], 32, repeats=3, k=5)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertGreater(row.search_ms, 0.0)
            self.assertGreater(row.build_ms, 0.0)
            self.assertGreater(row.filtered_ms, 0.0)
            self.assertGreater(row.searches_per_sec, 0.0)
        table = format_table(rows)
        self.assertIn("chunks", table)


if __name__ == "__main__":
    unittest.main()
