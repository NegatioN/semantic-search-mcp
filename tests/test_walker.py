import tempfile
import unittest
from pathlib import Path

from zemsearch import config as config_mod
from zemsearch.index.walker import walk


class TestWalker(unittest.TestCase):
    def test_excludes_binary_and_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "src").mkdir()
            (root / "src" / "a.py").write_text("print(1)")
            (root / "node_modules").mkdir()
            (root / "node_modules" / "x.js").write_text("var x = 1")
            (root / ".gitignore").write_text("ignored.txt\n")
            (root / "ignored.txt").write_text("hi")
            (root / "bin.dat").write_bytes(b"\x00\x01\x02\x03")
            (root / "empty.txt").write_text("")

            cfg = config_mod.load(root=d).index
            found = {p.relative_to(root).as_posix() for p in walk(root, cfg)}

            self.assertIn("src/a.py", found)
            self.assertNotIn("node_modules/x.js", found)
            self.assertNotIn("ignored.txt", found)
            self.assertNotIn("bin.dat", found)
            self.assertNotIn("empty.txt", found)


if __name__ == "__main__":
    unittest.main()
