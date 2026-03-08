import os
import httpx
from dotenv import load_dotenv

load_dotenv()

MAKEUP_PRODUCT_URL = os.getenv("MAKEUP_PRODUCT_URL") or os.getenv("MAKE_PRODUCT_URL") or "http://makeup-api.herokuapp.com/api/v1/products.json"


class Makeup:
    """Service 1: Makeup API – fetches products and returns transformed/non-verbatim output."""

    @staticmethod
    def get_makeup_products():
        response = httpx.get(MAKEUP_PRODUCT_URL, timeout=30.0)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def get_makeup_products_transformed(limit: int = 20):
        """Fetch products and return a summarized, natural-language description (not raw JSON)."""
        raw = Makeup.get_makeup_products()
        if not raw:
            return "No makeup products are available right now."
        # Limit size for context; summarize in a human-friendly way
        subset = raw[:limit] if isinstance(raw, list) else [raw]
        lines = []
        for p in subset:
            name = p.get("name") or "Unnamed"
            brand = p.get("brand") or "Unknown brand"
            ptype = p.get("product_type") or "product"
            price = p.get("price") or "—"
            desc = (p.get("description") or "")[:120]
            lines.append(f"• {name} ({brand}) — {ptype}, price {price}. {desc}")
        return "\n".join(lines) if lines else "No products found."
