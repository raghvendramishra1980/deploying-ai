# Makeup & Search Chat – Server

Backend and Gradio UI for the assignment chat: three services (Makeup API, semantic search, web search), guardrails, and a chat interface with memory.

## Services

1. **Makeup API (Service 1)**  
   Uses `http://makeup-api.herokuapp.com/api/v1/products.json`. Responses are turned into short, human-readable summaries (no raw JSON to the user).

2. **Semantic search (Service 2)**  
   ChromaDB with **file persistence** in `./chromadb_makeup/`.  
   - **Dataset**: makeup products from the same API (subset, kept under size limits).  
   - **Embedding process**: We use **sentence-transformers** (`gpt-4o-mini`). For each product we build a text from `name`, `description`, `product_type`, and `brand`. We encode these texts with the SentenceTransformer model and store the vectors in ChromaDB. The index is built on first use (or you can run a one-off script that calls `semantic_search.build_index_if_needed()`). No SQLite; ChromaDB uses its own persistence in the directory above.

3. **Web search (Service 3)**  
   Simple web search via **DuckDuckGo** (`duckduckgo-search`). Used for general or current-info queries when the model chooses to call the tool.

## Guardrails

- **Prompt**: Requests to reveal or change the system prompt are blocked with a short, safe message.  
- **Restricted topics**: The model is instructed not to answer questions about **cats/dogs**, **horoscopes/zodiac**, or **Taylor Swift**. Inputs that match these topics are blocked and the user is redirected to allowed topics (e.g. makeup, product search).

## Run

1. **Env**  
   Copy `.env.example` to `.env` and set at least:
   - `AWS_SECRET_ACCESS_KEY` – for the OpenAI-compatible API gateway.

2. **Install**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Backend**  
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Gradio UI** (after backend is up)  
   ```bash
   python gradio_app.py
   ```  
   Then open the URL shown (e.g. http://127.0.0.1:7860).

The chat uses the backend `/chat` endpoint, keeps conversation history in the Gradio session, and has a distinct “Glow” personality (warm, professional makeup/lifestyle assistant).
