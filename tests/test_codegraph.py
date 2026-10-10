import sqlite3
import tempfile
import unittest
from pathlib import Path

from zemsearch import config as config_mod
from zemsearch.codegraph import CodeGraphReader, db_fingerprint
from zemsearch.index.codegraph_indexer import CodeGraphIndexer
from zemsearch.index.store import SQLiteStore


def _node(nid, kind, name, qname, path, start=1, end=5, doc=None, sig=None):
    return {
        "id": nid,
        "kind": kind,
        "name": name,
        "qualified_name": qname,
        "file_path": path,
        "language": "go",
        "start_line": start,
        "end_line": end,
        "docstring": doc,
        "signature": sig,
        "return_type": None,
    }


def _write_db(path: Path, nodes, edges, state="complete", extraction="27"):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(str(path) + suffix).unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY, kind TEXT, name TEXT, qualified_name TEXT,
            file_path TEXT, language TEXT, start_line INT, end_line INT,
            docstring TEXT, signature TEXT, return_type TEXT
        );
        CREATE TABLE edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, target TEXT, kind TEXT
        );
        CREATE TABLE project_metadata (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    for n in nodes:
        conn.execute(
            "INSERT INTO nodes VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                n["id"], n["kind"], n["name"], n["qualified_name"], n["file_path"],
                n["language"], n["start_line"], n["end_line"], n["docstring"],
                n["signature"], n["return_type"],
            ),
        )
    for src, kind, tgt in edges:
        conn.execute("INSERT INTO edges(source, target, kind) VALUES(?,?,?)", (src, tgt, kind))
    conn.execute("INSERT INTO project_metadata VALUES('index_state', ?)", (state,))
    conn.execute(
        "INSERT INTO project_metadata VALUES('indexed_with_extraction_version', ?)", (extraction,)
    )
    conn.commit()
    conn.close()


def _base_nodes():
    return [
        _node("fnA", "function", "Alpha", "Alpha", "a.go", doc="does alpha", sig="() int"),
        _node("fnB", "function", "Beta", "Beta", "b.go", doc="does beta", sig="() int"),
    ]


class FakeClient:
    def __init__(self):
        self.batches: list[list[str]] = []

    def embed(self, texts):
        self.batches.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


class TestCodeGraphReader(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = self.root / ".codegraph/codegraph.db"
        (self.root / "a.go").write_text("func Alpha() int { return Beta() }")
        (self.root / "b.go").write_text("func Beta() int { return 1 }")
        _write_db(self.db, _base_nodes(), [("fnA", "calls", "fnB")])

    def tearDown(self):
        self._tmp.cleanup()

    def _reader(self):
        return CodeGraphReader(self.db, self.root)

    def test_enriched_text_has_signature_doc_and_neighbours(self):
        chunks = {c.symbol: c for c in self._reader().chunks()}
        self.assertEqual(set(chunks), {"Alpha", "Beta"})
        alpha = chunks["Alpha"].text
        self.assertIn("function Alpha", alpha)
        self.assertIn("signature: Alpha() int", alpha)
        self.assertIn("doc: does alpha", alpha)
        self.assertIn("calls: Beta", alpha)
        self.assertIn("return Beta()", alpha)  # source slice folded in
        self.assertEqual(chunks["Alpha"].granularity, "symbol")
        self.assertEqual(chunks["Beta"].title, "b.go::Beta")

    def test_embeds_declaration_kinds_but_not_structural(self):
        _write_db(
            self.db,
            [
                _node("c1", "class", "Foo", "Foo", "a.scala"),
                _node("m1", "method", "bar", "Foo.bar", "a.scala"),
                _node("t1", "trait", "Bar", "Bar", "a.scala"),
                _node("imp", "import", "zio.http", "zio.http", "a.scala"),
                _node("f1", "file", "a.scala", "a.scala", "a.scala"),
            ],
            [],
        )
        chunks = {c.symbol for c in self._reader().chunks()}
        self.assertEqual(chunks, {"Foo", "Foo.bar", "Bar"})

    def test_not_ready_when_index_state_incomplete(self):
        _write_db(self.db, _base_nodes(), [], state="indexing")
        self.assertFalse(self._reader().is_ready())

    def test_fingerprint_changes_on_write(self):
        before = db_fingerprint(self.db)
        _write_db(self.db, _base_nodes(), [("fnA", "calls", "fnB")])
        self.assertNotEqual(before, db_fingerprint(self.db))


class TestCodeGraphIndexer(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = self.root / ".codegraph/codegraph.db"
        (self.root / "a.go").write_text("func Alpha() int { return Beta() }")
        (self.root / "b.go").write_text("func Beta() int { return 1 }")
        _write_db(self.db, _base_nodes(), [("fnA", "calls", "fnB")])
        self.store = SQLiteStore(self.root / ".zemsearch/index.db")
        self.store.init_schema()
        self.client = FakeClient()
        cfg = config_mod.load(root=str(self.root))
        cfg.index.source = "codegraph"
        self.cfg = cfg.index
        self.indexer = CodeGraphIndexer(self.root, self.cfg, self.client, self.store)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_initial_index(self):
        report = self.indexer.index()
        self.assertEqual(report.changed, 2)
        self.assertEqual(report.chunks, 2)
        self.assertEqual(report.scanned, 2)
        self.assertEqual(self.store.stats()["chunks"], 2)
        self.assertEqual(len(self.store.get_symbol_hashes()), 2)

    def test_second_run_is_noop(self):
        self.indexer.index()
        self.client.batches.clear()
        report = self.indexer.index()
        self.assertEqual(report.changed, 0)
        self.assertEqual(self.client.batches, [])

    def test_progress_reports_cumulative_counts(self):
        seen = []
        self.indexer.index(
            progress=lambda path, chunks, done, total: seen.append((path, chunks, done, total))
        )
        self.assertEqual([s[0] for s in seen], ["a.go", "b.go"])
        self.assertEqual([s[3] for s in seen], [2, 2])  # total is constant
        self.assertEqual([s[2] for s in seen], [1, 2])  # done accumulates

    def test_changed_symbol_reembeds_only_its_file(self):
        self.indexer.index()
        self.client.batches.clear()
        _write_db(
            self.db,
            [
                _node("fnA", "function", "Alpha", "Alpha", "a.go", doc="does alpha v2", sig="() int"),
                _node("fnB", "function", "Beta", "Beta", "b.go", doc="does beta", sig="() int"),
            ],
            [("fnA", "calls", "fnB")],
        )
        report = self.indexer.index()
        self.assertEqual(report.changed, 1)
        self.assertEqual(len(self.client.batches[-1]), 1)  # only a.go re-embedded

    def test_deleted_symbol_is_removed(self):
        self.indexer.index()
        _write_db(
            self.db,
            [_node("fnB", "function", "Beta", "Beta", "b.go", doc="does beta", sig="() int")],
            [],
        )
        report = self.indexer.index()
        self.assertEqual(report.deleted, 1)
        self.assertEqual(self.store.stats()["chunks"], 1)
        self.assertEqual({k[1] for k in self.store.get_symbol_hashes()}, {"Beta"})

    def test_neighbour_rename_dirties_caller(self):
        self.indexer.index()
        self.client.batches.clear()
        # Beta is renamed (qualified_name unchanged); Alpha's "calls" line changes.
        _write_db(
            self.db,
            [
                _node("fnA", "function", "Alpha", "Alpha", "a.go", doc="does alpha", sig="() int"),
                _node("fnB", "function", "Bee", "Beta", "b.go", doc="does beta", sig="() int"),
            ],
            [("fnA", "calls", "fnB")],
        )
        report = self.indexer.index()
        self.assertEqual(report.changed, 2)
        alpha_text = next(
            c for c in self.indexer.reader.chunks() if c.symbol == "Alpha"
        ).text
        self.assertIn("calls: Bee", alpha_text)

    def test_skips_when_graph_not_ready(self):
        _write_db(self.db, _base_nodes(), [("fnA", "calls", "fnB")], state="indexing")
        report = self.indexer.index()
        self.assertEqual(report.changed, 0)
        self.assertEqual(report.chunks, 0)
        self.assertEqual(self.client.batches, [])


if __name__ == "__main__":
    unittest.main()
