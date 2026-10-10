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

    def test_expanded_default_dirs_are_pruned(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for sub in [".gradle", "vendor", "coverage", ".nuxt", "obj", ".cache", "Pods"]:
                (root / sub).mkdir()
                (root / sub / "f.txt").write_text("x")
            (root / "src").mkdir()
            (root / "src" / "a.py").write_text("print(1)")

            cfg = config_mod.load(root=d).index
            found = {p.relative_to(root).as_posix() for p in walk(root, cfg)}
            self.assertEqual(found, {"src/a.py"})

    def test_gitignore_negation_reincludes_a_default(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "vendor").mkdir()
            (root / "vendor" / "keep.go").write_text("package vendored")
            (root / ".gitignore").write_text("!vendor/\n")

            cfg = config_mod.load(root=d).index
            found = {p.relative_to(root).as_posix() for p in walk(root, cfg)}
            self.assertIn("vendor/keep.go", found)

    def test_nested_gitignore_is_scoped_and_honors_negation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "sub" / "deep").mkdir(parents=True)
            (root / "sub" / ".gitignore").write_text("*.log\n!keep.log\n")
            (root / "sub" / "a.log").write_text("x")
            (root / "sub" / "deep" / "b.log").write_text("x")
            (root / "sub" / "keep.log").write_text("x")
            (root / "sub" / "main.py").write_text("x")
            (root / "top.log").write_text("x")

            cfg = config_mod.load(root=d).index
            found = {p.relative_to(root).as_posix() for p in walk(root, cfg)}
            self.assertNotIn("sub/a.log", found)       # bare pattern matches here
            self.assertNotIn("sub/deep/b.log", found)  # ...and any depth below
            self.assertIn("sub/keep.log", found)       # negation wins
            self.assertIn("sub/main.py", found)
            self.assertIn("top.log", found)            # sub scope doesn't leak up

    def test_jvm_build_dir_stays_indexed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "src/main/java/com/x/build").mkdir(parents=True)
            (root / "src/main/java/com/x/build/Foo.java").write_text("class Foo {}")
            (root / "build").mkdir()
            (root / "build" / "out.txt").write_text("x")

            cfg = config_mod.load(root=d).index
            found = {p.relative_to(root).as_posix() for p in walk(root, cfg)}
            self.assertIn("src/main/java/com/x/build/Foo.java", found)
            self.assertNotIn("build/out.txt", found)

    def test_git_info_exclude_is_honored(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".git" / "info").mkdir(parents=True)
            (root / ".git" / "info" / "exclude").write_text("secret.txt\n")
            (root / "secret.txt").write_text("x")
            (root / "public.txt").write_text("x")

            cfg = config_mod.load(root=d).index
            found = {p.relative_to(root).as_posix() for p in walk(root, cfg)}
            self.assertNotIn("secret.txt", found)
            self.assertIn("public.txt", found)

    def test_respect_gitignore_false_still_applies_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".gitignore").write_text("ignored.txt\n")
            (root / "ignored.txt").write_text("x")
            (root / "node_modules").mkdir()
            (root / "node_modules" / "y.js").write_text("x")

            cfg = config_mod.load(root=d).index
            cfg.respect_gitignore = False
            found = {p.relative_to(root).as_posix() for p in walk(root, cfg)}
            self.assertIn("ignored.txt", found)          # gitignore not applied
            self.assertNotIn("node_modules/y.js", found)  # defaults still applied

    def test_malformed_gitignore_line_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".gitignore").write_text("!\nvalid_ignore.txt\n")
            (root / "valid_ignore.txt").write_text("x")
            (root / "keep.txt").write_text("x")

            cfg = config_mod.load(root=d).index
            found = {p.relative_to(root).as_posix() for p in walk(root, cfg)}
            self.assertNotIn("valid_ignore.txt", found)
            self.assertIn("keep.txt", found)


if __name__ == "__main__":
    unittest.main()
