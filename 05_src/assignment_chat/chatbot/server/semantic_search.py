"""
Service 2: Semantic search over makeup products using ChromaDB with file persistence.
Dataset: makeup API products (stored in Chroma). Embedding process described in README.
"""

from pathlib import Path

import chromadb
from chromadb.config import Settings

from makeupApi import Makeup

# Lazy-load sentence_transformers to avoid pulling in PyTorch/NumPy at import time
# (avoids version conflicts so the server can start even when torch/transformers are broken)
_SentenceTransformer = None


def _get_sentence_transformer():
    """Import sentence_transformers only when needed; return None if import fails."""
    global _SentenceTransformer
    if _SentenceTransformer is not None:
        return _SentenceTransformer
    try:
        from sentence_transformers import SentenceTransformer as ST
        _SentenceTransformer = ST
        return ST
    except Exception:
        _SentenceTransformer = False
        return None

# Persist under server directory so it's easy to ship (under 40 MB with makeup data)
CHROMA_DIR = Path(__file__).resolve().parent / "chromadb_makeup"
COLLECTION_NAME = "makeup_products"


def get_chroma_client():
    """ChromaDB client with file persistence."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def build_index_if_needed():
    """Build and persist the makeup products index. Safe to call on startup or first query."""
    client = get_chroma_client()
    try:
        coll = client.get_collection(name=COLLECTION_NAME)
        if coll.count() > 0:
            return  # already built
    except Exception:
        pass

    ST = _get_sentence_transformer()
    if ST is None:
        return  # embedding lib unavailable; skip building index

    products = Makeup.get_makeup_products()
    if not products:
        return
    if not isinstance(products, list):
        products = [products]

    texts = []
    ids = []
    metadatas = []
    for i, p in enumerate(products[:500]):  # cap for size/speed
        text = f"{p.get('name', '')} {p.get('description', '')} {p.get('product_type', '')} {p.get('brand', '')}"
        texts.append(text.strip() or str(i))
        ids.append(str(p.get("id", i)))
        metadatas.append({
            "brand": str(p.get("brand") or ""),
            "price": str(p.get("price") or ""),
            "product_type": str(p.get("product_type") or ""),
            "name": str(p.get("name") or ""),
        })

    embedder = ST("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts).tolist()
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"description": "makeup products"})
    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)


def semantic_search(query: str, n_results: int = 5):
    """
    Run semantic search over makeup products. Builds index on first use if needed.
    Returns list of dicts with document, metadata, distance.
    """
    build_index_if_needed()
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return []

    ST = _get_sentence_transformer()
    if ST is not None:
        embedder = ST("all-MiniLM-L6-v2")
        q_emb = embedder.encode([query]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=n_results)
    else:
        try:
            results = collection.query(query_texts=[query], n_results=n_results)
        except Exception:
            return []

    out = []
    if results and results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            meta = (results["metadatas"][0][i]) if results.get("metadatas") and results["metadatas"][0] else {}
            out.append({
                "document": doc,
                "metadata": meta,
                "distance": (results["distances"][0][i]) if results.get("distances") and results["distances"][0] else None,
            })
    return out
