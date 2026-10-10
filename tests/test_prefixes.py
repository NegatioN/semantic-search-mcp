import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zemsearch.model.prefixes import format_document, format_query


class TestCodeRetrievalPrefixes(unittest.TestCase):
    def test_query_prefix_exact(self):
        self.assertEqual(
            format_query("find the config loader"),
            "task: code retrieval | query: find the config loader",
        )

    def test_document_prefix_with_title(self):
        self.assertEqual(
            format_document("def f():\n    pass", title="src/a.py"),
            "title: src/a.py | text: def f():\n    pass",
        )

    def test_document_prefix_symbol_title(self):
        self.assertEqual(
            format_document("x = 1", title="src/a.py::f"),
            "title: src/a.py::f | text: x = 1",
        )

    def test_document_without_title_uses_none(self):
        self.assertEqual(format_document("body"), "title: none | text: body")
        self.assertEqual(format_document("body", title=""), "title: none | text: body")


if __name__ == "__main__":
    unittest.main()
