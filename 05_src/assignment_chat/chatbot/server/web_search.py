"""
Service 3: Simple web search (DuckDuckGo). Used for general queries when appropriate.
"""

from typing import List, Dict, Any

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False


def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Perform a simple web search and return titles, snippets, and URLs.
    Returns empty list if duckduckgo_search is not installed or on error.
    """
    if not HAS_DDGS:
        return []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        out = []
        for r in results:
            out.append({
                "title": r.get("title", ""),
                "body": r.get("body", ""),
                "href": r.get("href", ""),
            })
        return out
    except Exception:
        return []
