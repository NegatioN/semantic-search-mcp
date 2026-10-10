import tempfile
import threading
import time
import unittest
from pathlib import Path

from zemsearch import config as config_mod
from zemsearch.index.walker import build_scope
from zemsearch.index.watcher import ReindexScheduler, should_ignore


class TestShouldIgnore(unittest.TestCase):
    def test_ignored(self):
        for path in [
            ".git/config",
            ".jj/repo",
            "node_modules/pkg/index.js",
            "src/.zemsearch/index.db",
            "foo/bar.py.swp",
            "x/.#lock",
            "editor~",
            ".DS_Store",
        ]:
            self.assertTrue(should_ignore(path), path)

    def test_kept(self):
        for path in ["src/a.py", "README.md", "zemsearch.toml", "pkg/main.rs"]:
            self.assertFalse(should_ignore(path), path)

    def test_store_path_ignored(self):
        self.assertTrue(should_ignore("/tmp/ws/index.db", Path("/tmp/ws/index.db")))

    def test_scope_honors_gitignore(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".gitignore").write_text("generated/\n")
            (root / "generated").mkdir()
            (root / "generated" / "x.py").write_text("x")
            (root / "src.py").write_text("x")
            scope = build_scope(root, config_mod.load(root=d).index)
            self.assertTrue(should_ignore(str(root / "generated" / "x.py"), scope=scope))
            self.assertFalse(should_ignore(str(root / "src.py"), scope=scope))


class _Recorder:
    def __init__(self):
        self.calls = 0
        self.event = threading.Event()

    def __call__(self):
        self.calls += 1
        self.event.set()


class TestReindexScheduler(unittest.TestCase):
    def test_event_triggers_debounced_reindex(self):
        with tempfile.TemporaryDirectory() as d:
            rec = _Recorder()
            scheduler = ReindexScheduler([Path(d)], rec, interval_seconds=0, debounce_seconds=0.1)
            scheduler.start()
            try:
                (Path(d) / "a.py").write_text("print(1)\n")
                self.assertTrue(rec.event.wait(5.0), "reindex was not triggered")
            finally:
                scheduler.stop()
            self.assertGreaterEqual(rec.calls, 1)

    def test_periodic_reindex_without_events(self):
        with tempfile.TemporaryDirectory() as d:
            rec = _Recorder()
            scheduler = ReindexScheduler([Path(d)], rec, interval_seconds=0.2, debounce_seconds=0.1)
            scheduler.start()
            try:
                self.assertTrue(rec.event.wait(5.0), "periodic reindex did not fire")
            finally:
                scheduler.stop()
            self.assertGreaterEqual(rec.calls, 1)

    def test_ignored_changes_do_not_trigger(self):
        with tempfile.TemporaryDirectory() as d:
            rec = _Recorder()
            scheduler = ReindexScheduler([Path(d)], rec, interval_seconds=0, debounce_seconds=0.1)
            scheduler.start()
            try:
                (Path(d) / "node_modules").mkdir()
                (Path(d) / "node_modules" / "x.js").write_text("var x = 1\n")
                time.sleep(0.6)
                self.assertEqual(rec.calls, 0)
            finally:
                scheduler.stop()

    def test_running_and_stop(self):
        with tempfile.TemporaryDirectory() as d:
            scheduler = ReindexScheduler(
                [Path(d)], _Recorder(), interval_seconds=0, debounce_seconds=0.1
            )
            scheduler.start()
            self.assertTrue(scheduler.running)
            scheduler.stop()
            self.assertFalse(scheduler.running)


if __name__ == "__main__":
    unittest.main()
