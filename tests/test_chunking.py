import unittest

from gemma_embedder.chunking.base import Chunk
from gemma_embedder.chunking.file import split_text


class TestSplitText(unittest.TestCase):
    def test_small_text_single_window(self):
        windows = split_text("a\nb\nc\n", max_chars=100)
        self.assertEqual(len(windows), 1)
        start, _end, content = windows[0]
        self.assertEqual(start, 1)
        self.assertEqual(content, "a\nb\nc\n")

    def test_large_text_multiple_windows(self):
        text = "".join(f"line {i}\n" for i in range(500))
        windows = split_text(text, max_chars=200, overlap_chars=50)
        self.assertGreater(len(windows), 1)
        # windows cover all lines, start at 1 and end at 500
        self.assertEqual(windows[0][0], 1)
        self.assertEqual(windows[-1][1], 500)
        for start, end, _ in windows:
            self.assertLessEqual(start, end)

    def test_windows_overlap(self):
        text = "".join(f"line {i}\n" for i in range(500))
        windows = split_text(text, max_chars=200, overlap_chars=50)
        self.assertGreater(windows[1][0], windows[0][0])


class TestChunkDocument(unittest.TestCase):
    def test_document_uses_title_prefix(self):
        chunk = Chunk(path="a/b.py", title="a/b.py", text="x = 1")
        self.assertEqual(chunk.document(), "title: a/b.py | text: x = 1")


if __name__ == "__main__":
    unittest.main()
