import tempfile
import unittest

from gemma_embedder import config as config_mod
from gemma_embedder.mcp.server import build_server
from gemma_embedder.runtime import Runtime


class TestMcpServer(unittest.TestCase):
    def test_build_server_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = config_mod.load(root=d)
            runtime = Runtime(cfg)
            try:
                server = build_server(runtime)
                self.assertIsNotNone(server)
                self.assertEqual(server.name, "gemma-embedder")
            finally:
                runtime.close()


if __name__ == "__main__":
    unittest.main()
