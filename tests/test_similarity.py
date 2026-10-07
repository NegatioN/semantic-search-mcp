import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gemma_embedder.model.client import (
    cosine,
    dot,
    l2_normalize,
    view,
)
from gemma_embedder.probe import parse_host_port


class TestVectorMath(unittest.TestCase):
    def test_l2_normalize_unit_norm(self):
        v = l2_normalize([3.0, 4.0])
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in v)), 1.0)

    def test_l2_normalize_zero_vector(self):
        self.assertEqual(l2_normalize([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0])

    def test_dot_equals_cosine_for_unit_vectors(self):
        a = l2_normalize([1.0, 2.0, 3.0])
        b = l2_normalize([3.0, 1.0, 0.0])
        self.assertAlmostEqual(dot(a, b), cosine(a, b), places=9)

    def test_view_truncates_then_renormalizes(self):
        out = view([3.0, 4.0, 99.0], 2)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0], 0.6)
        self.assertAlmostEqual(out[1], 0.8)

    def test_view_at_native_equals_normalize(self):
        v = [0.1] * 768
        self.assertEqual(view(v, 768), l2_normalize(v))

    def test_view_dim_too_large(self):
        with self.assertRaises(ValueError):
            view([1.0, 2.0], 3)


class TestHostPortParsing(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(parse_host_port("http://127.0.0.1:8080"), ("127.0.0.1", 8080))

    def test_custom_port(self):
        self.assertEqual(parse_host_port("http://localhost:9090"), ("localhost", 9090))

    def test_no_port_uses_default(self):
        self.assertEqual(parse_host_port("http://localhost"), ("localhost", 8080))


if __name__ == "__main__":
    unittest.main()
