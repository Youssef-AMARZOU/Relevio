"""MCP server tests: tool registration and placeholder-mode wiring.

Self-skips when the mcp SDK or the serving stack is absent, matching the
bare-stdlib CI constraint of the rest of the suite.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import unittest

_SERVING_DEPS = ("fastapi", "numpy", "pandas", "pyarrow", "lightgbm")
_HAS_SERVING = all(importlib.util.find_spec(name) is not None for name in _SERVING_DEPS)
_HAS_MCP = importlib.util.find_spec("mcp") is not None

if _HAS_MCP and _HAS_SERVING:
    from otto_rec.serving import mcp_server

_ENV_KEYS = ("SERVING_PIPELINE", "SERVING_ANN", "REDIS_URL")


@unittest.skipUnless(_HAS_MCP and _HAS_SERVING, "mcp + serving deps not installed")
class McpServerTest(unittest.TestCase):
    """Placeholder mode keeps every tool answering without artifacts."""

    def setUp(self) -> None:
        self._saved = {key: os.environ.pop(key, None) for key in _ENV_KEYS}
        os.environ["SERVING_PIPELINE"] = "off"
        os.environ["SERVING_ANN"] = "off"
        mcp_server._ready = False

    def tearDown(self) -> None:
        mcp_server._ready = False
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_registered_tools(self) -> None:
        tools = asyncio.run(mcp_server.mcp.list_tools())
        names = {tool.name for tool in tools}
        self.assertEqual(names, {"recommend", "search", "explain", "health"})

    def test_recommend_placeholder(self) -> None:
        response = mcp_server.recommend("s1", 5, ["1001"], None)
        self.assertEqual(response["session_id"], "s1")
        self.assertEqual(len(response["candidates"]), 5)
        self.assertTrue(all(c["item_id"] != "1001" for c in response["candidates"]))
        self.assertTrue(all(c["sources"] == ["placeholder"] for c in response["candidates"]))
        self.assertIn("model_version", response)

    def test_recommend_events_override_history(self) -> None:
        events = [{"item_id": "1002", "etype": "orders", "ts": 1664500000000}]
        response = mcp_server.recommend("s2", 3, ["1001"], events)
        self.assertEqual(len(response["candidates"]), 3)
        self.assertNotIn("1002", [c["item_id"] for c in response["candidates"]])

    def test_search_placeholder(self) -> None:
        response = mcp_server.search("running shoes", 7, [], None)
        self.assertEqual(response["query"], "running shoes")
        self.assertEqual(len(response["results"]), 7)

    def test_explain_without_pipeline(self) -> None:
        response = mcp_server.explain("s1", "1005", ["1001"], None)
        self.assertIn("Pipeline not loaded", response["note"])
        self.assertEqual(response["shap_values"], {})

    def test_health(self) -> None:
        response = mcp_server.health()
        self.assertEqual(response["status"], "ok")
        self.assertFalse(response["pipeline"])
        self.assertIn("model_version", response)
