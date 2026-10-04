"""MCP server: Relevio recommendations, search and explanations over stdio.

Exposes the exact handlers of the FastAPI app (``otto_rec.serving.app``) so
an MCP client - Claude Desktop, Claude Code, Cursor, the Glama Inspector -
gets the identical two-stage pipeline, Redis digest cache and deterministic
placeholder fallbacks. Artifacts load lazily on the first tool call through
``init_state`` with the same ``SERVING_PIPELINE`` / ``SERVING_ANN`` /
``PROCESSED_DIR`` / ``MODEL_PATH`` environment variables as the HTTP
service, so a fresh clone answers in placeholder mode instead of failing.

Run from the repo root (launcher adds ``src/`` to ``sys.path``):

    python mcp_server.py
    # equivalent: PYTHONPATH=src python -m otto_rec.serving.mcp_server
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from otto_rec.serving.app import Event
from otto_rec.serving.app import ExplainRequest as ExplainHTTPRequest
from otto_rec.serving.app import RecommendRequest as RecommendHTTPRequest
from otto_rec.serving.app import SearchRequest as SearchHTTPRequest
from otto_rec.serving.app import app
from otto_rec.serving.app import explain as explain_handler
from otto_rec.serving.app import health as health_handler
from otto_rec.serving.app import init_state
from otto_rec.serving.app import recommend as recommend_handler
from otto_rec.serving.app import search as search_handler

SERVER_NAME = "Relevio"
SERVER_VERSION = "1.0.0"

mcp = MCPServer(
    name=SERVER_NAME,
    version=SERVER_VERSION,
    description="Two-stage product recommendations, session search and score explanations over the OTTO demo artifacts.",
    instructions=(
        "Relevio ranks products for a browsing session: co-visitation and "
        "popularity candidates (plus dense ANN extras when available) are "
        "scored by a LambdaMART ranker. Pass the session as plain item ids "
        "in 'history', or as typed events (clicks/carts/orders with epoch-ms "
        "timestamps) in 'events' - events win when both are present. "
        "Recommendations never repeat history items."
    ),
)

_ready = False


def _ensure_ready() -> None:
    """Load the serving artifacts once, on first use (placeholder if absent)."""
    global _ready
    if not _ready:
        init_state(app.state)
        _ready = True


@mcp.tool()
def recommend(
    session_id: str,
    k: int = 20,
    history: list[str] | None = None,
    events: list[Event] | None = None,
) -> dict:
    """Rank the top-k products for a browsing session.

    Two-stage pipeline: co-visitation top-100 union per-type popularity pool
    (plus dense ANN candidates when loaded), scored by the LambdaMART
    ranker; session history items are excluded from the output. Falls back
    to deterministic placeholders when artifacts are absent.

    Args:
        session_id: Arbitrary id echoed back in the response.
        k: Number of candidates to return (1-100).
        history: Session item ids (plain click history).
        events: Typed session events - {"item_id", "etype":
            "clicks"|"carts"|"orders", "ts": epoch-ms}; preferred over history.
    """
    _ensure_ready()
    request = RecommendHTTPRequest(session_id=session_id, k=k, history=history or [], events=events)
    return recommend_handler(request).model_dump()


@mcp.tool()
def search(
    query: str,
    k: int = 20,
    history: list[str] | None = None,
    events: list[Event] | None = None,
) -> dict:
    """Retrieve products similar to the session (dense two-tower search).

    OTTO has no text corpus, so 'query' is echoed for context while
    retrieval runs over the session history: FAISS two-tower hits when the
    M2 artifacts are present, a deterministic placeholder otherwise.

    Args:
        query: Free-text query, kept in the response for context.
        k: Number of results to return (1-100).
        history: Session item ids driving the retrieval.
        events: Typed session events; preferred over history.
    """
    _ensure_ready()
    request = SearchHTTPRequest(query=query, k=k, history=history or [], events=events)
    return search_handler(request).model_dump()


@mcp.tool()
def explain(
    session_id: str,
    item_id: str,
    history: list[str] | None = None,
    events: list[Event] | None = None,
) -> dict:
    """Explain why the ranker scored one item for this session (TreeSHAP).

    Returns per-feature SHAP attributions (popularity, co-visitation,
    session features), the base value and the raw LambdaMART score. Needs
    the trained ranker artifacts; returns an explanatory note in
    placeholder mode. First call pays a ~2.5 s shap import, ~35 ms warm.

    Args:
        session_id: Arbitrary id echoed back in the response.
        item_id: The candidate item to attribute (must be a known item).
        history: Session item ids.
        events: Typed session events; preferred over history.
    """
    _ensure_ready()
    request = ExplainHTTPRequest(session_id=session_id, item_id=item_id, history=history or [], events=events)
    return explain_handler(request).model_dump()


@mcp.tool()
def health() -> dict:
    """Service status: model version and pipeline / ANN / Redis availability.

    'pipeline' is true when the two-stage artifacts are loaded, false in
    placeholder mode (fresh clone without data).
    """
    _ensure_ready()
    return health_handler().model_dump()


def main() -> None:
    """Run the MCP server on stdio (blocks until the client disconnects)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
