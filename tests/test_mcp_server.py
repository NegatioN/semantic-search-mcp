import tempfile
import unittest

from zemsearch import config as config_mod
from zemsearch.mcp.server import build_server
from zemsearch.runtime import Runtime


class TestMcpServer(unittest.TestCase):
    def test_build_server_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = config_mod.load(root=d)
            runtime = Runtime(cfg)
            try:
                server = build_server(runtime)
                self.assertIsNotNone(server)
                self.assertEqual(server.name, "zemsearch")
            finally:
                runtime.close()


if __name__ == "__main__":
    unittest.main()
