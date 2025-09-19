# AI Nutrition App

Production-ready RAG-powered nutrition assistant with a FastAPI backend and Streamlit front end. The backend builds a retrieval pipeline that indexes heterogeneous nutrition documents, enriches metadata, and uses OpenAI (plus optional Cohere) models to generate evidence-based responses. The Streamlit UI lets users upload new notes and ask questions backed by their curated knowledge base.

## Architecture
- **FastAPI** (`backend/app`): hosts ingestion and question-answering endpoints, manages hybrid dense/keyword retrieval, optional reranking, and vector store backends (FAISS or pgvector).
- **Retrieval pipeline**: token-aware chunking (tiktoken-based) with section-aware metadata, dense embeddings (OpenAI) plus BM25 keyword search, optional Cohere/Local reranker, and configurable ensemble weighting.
- **Document loaders**: text/markdown, PDF (via `pypdf`), DOCX (`python-docx`), and web pages (`trafilatura`). Uploads share rich metadata including title, section, chunk indices, and page numbers.
- **Vector stores**: default FAISS on disk; optional managed Postgres (`pgvector`) backend for multi-instance deployments.
- **Streamlit** (`frontend/app.py`): lightweight client for uploading nutrition references, ingesting URLs, and chatting with the assistant. Retrieved context displays section, page, and chunk ordering for transparency.

## Prerequisites
- Python 3.10+
- OpenAI API key with access to `gpt-3.5-turbo` (or update the model names in `.env`).
- (Optional) Cohere API key for best-quality reranking.
- (Optional) Postgres database with the `pgvector` extension enabled if you opt into managed storage.

## Setup

1. **Clone repo & create virtual environments**
   ```bash
   python -m venv .venv-backend
   source .venv-backend/bin/activate
   pip install -r backend/requirements.txt
   ```

   For the Streamlit client:
   ```bash
   python -m venv .venv-frontend
   source .venv-frontend/bin/activate
   pip install -r frontend/requirements.txt
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   # edit .env and set OPENAI_API_KEY=...
   ```
   The backend automatically loads `.env`. Adjust feature flags or model names as needed.

### Notable environment toggles
- `ENABLE_HYBRID=true` – blend dense embeddings with BM25 keyword search (default on).
- `HYBRID_ALPHA=0.6` – weighting between dense (alpha) and keyword (1-alpha) scores.
- `ENABLE_RERANK=true` / `RERANK_PROVIDER=cohere|local|none` – rerank candidates with Cohere ReRank when available or a lightweight local heuristic fallback.
- `VECTOR_BACKEND=faiss|pgvector` – select the vector store. When using `pgvector`, set `POSTGRES_DSN=postgresql+psycopg2://user:pass@host:5432/db`.
- `ENABLE_MEMORY=true` – persist conversational state in SQLite (default `backend/memory.db`) or Postgres; configure with `MEMORY_BACKEND` / `MEMORY_DSN`.
- `HISTORY_WINDOW`, `SUMMARY_EVERY_N_TURNS`, `SUMMARY_MAX_TOKENS` – tune how much prior conversation is injected via the summary buffer.
- `CHUNKER=token|char`, `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS` – configure the token-aware splitter.
- Loader toggles (`ENABLE_PDF_LOADER`, `ENABLE_DOCX_LOADER`, `ENABLE_WEB_LOADER`) let you opt out of specific parsers.

## Running the stack

1. **Start the FastAPI server**
   ```bash
   source .venv-backend/bin/activate
   cd backend
   uvicorn app.main:app --reload
   ```

2. **Launch the Streamlit UI**
   ```bash
   source .venv-frontend/bin/activate
   streamlit run frontend/app.py
   ```

   The sidebar contains a "Ping backend" button; ensure it reports a healthy connection.

## Using the app
- Seed documents by placing `.txt`, `.md`, `.pdf`, `.docx`, or `.url` link files inside `backend/data/`. Use the Streamlit uploader or `/api/ingest_url` for runtime additions.
- Use the session endpoints (`/api/sessions/...`) to create and manage persistent conversations. Pass `session_id` (and optionally `user_id`) to `/api/ask` to enable follow-up questions that remember prior context.
- The Streamlit sidebar now exposes "New conversation" and session selection so you can manage threads directly from the UI. Set an optional user ID to scope sessions per user.
- Ask targeted questions (e.g., "How much protein for marathon training?") and review the retrieved context. Each chunk shows the originating section, page (if applicable), and chunk index for traceability.
- Hybrid retrieval and reranking can be toggled via env variables—experiment to balance speed vs. quality.
- The backend persists the FAISS index under `backend/vector_store/`; delete that folder (or switch `VECTOR_BACKEND`) to rebuild from scratch.

## Testing and validation
- Verify the API is alive: `curl http://localhost:8000/api/health`.
- Ask a question before and after ingesting new PDFs/DOCX files to observe enriched metadata in the Streamlit context panel.
- Toggle `ENABLE_HYBRID` / `ENABLE_RERANK` and compare `context` ordering from `/api/ask`.
- If using Postgres, confirm persistence across restarts by pointing multiple backend instances at the same DSN.
- Test conversational memory: `POST /api/sessions/create`, call `/api/ask` with the returned `session_id` for multi-turn queries, then inspect `/api/sessions/{session_id}/history` after a restart.
- Run automated checks (unit + integration): `./scripts/run_tests.sh`

## Customization tips
- Swap the OpenAI or Cohere models via environment overrides (e.g., GPT-4o, Cohere `rerank-3`).
- Extend `RAGService` with new loaders (e.g., CSV meal logs) or analytics endpoints in `backend/app/api/`.
- Enhance the Streamlit UI with macro calculators, progress tracking, or integrations to nutrition trackers.
