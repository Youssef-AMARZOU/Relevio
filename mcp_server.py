"""Launcher for the Relevio MCP server (stdio transport).

Usage: python mcp_server.py
Adds src/ to sys.path so no package install is required; see
otto_rec.serving.mcp_server for the tool implementations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from otto_rec.serving.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    main()
