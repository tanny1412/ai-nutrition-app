# AI Nutrition Assistant

An end‑to‑end nutrition chatbot with Retrieval‑Augmented Generation (RAG), per‑user memory, and clean FastAPI endpoints. It supports hybrid retrieval (dense vectors + BM25), optional reranking, and flexible, LLM‑driven personal memory.

Features
- Hybrid RAG: FAISS dense vectors (OpenAI embeddings) + BM25 keyword search with score ensembling.
- Optional reranking: Cohere reranker or a lightweight local reranker.
- LLM‑driven memory: Stores and recalls personal facts per user. Flexible extraction and answering from memory.
- Conversation persistence: Per‑session summaries and history with SQLite/Postgres via SQLAlchemy.
- Data ingestion: Text/Markdown/PDF/DOCX and URLs, with configurable loaders.
- Config‑first: All major behaviors are controlled through environment variables.

Architecture
- Backend
  - FastAPI app with routes for asking questions, ingesting content, and managing sessions.
  - RAG Service: Chunking, vector/keyword retrieval, hybrid ensembling, and context construction.
  - Memory Service: SQL‑backed conversations, summaries, and user key‑value memory.
  - Agent Pipelines:
    - SimpleAgenticPipeline: LangChain conversation memory per user (in‑process) + RAG enhancement.
    - AgenticPipeline: Graph‑based router with LLM‑driven personal memory extraction/recall and RAG/chitchat routes.
- Frontend
  - Minimal example app in `frontend/` (you can wire to your UI of choice).

Key Files
- Backend API routes: `backend/app/api/routes.py`
- RAG service: `backend/app/services/rag_service.py`
- Persistent memory service: `backend/app/services/memory_service.py`
- Simple pipeline: `backend/app/services/simple_agentic_pipeline.py`
- Advanced agentic pipeline: `backend/app/services/agentic_pipeline.py`
- Settings: `backend/app/config.py`

RAG Overview
- Dense vectors: FAISS (local, persisted under `backend/vector_store`) or pgvector (Postgres).
- Keyword index: BM25 (rank_bm25) for lexical matching.
- Hybrid: Scores from dense and BM25 are normalized and combined (see `ENABLE_HYBRID` and `HYBRID_ALPHA`).
- Reranking (optional): Cohere (`RERANK_PROVIDER=cohere`) or local heuristic (`RERANK_PROVIDER=local`).
- Context assembly: Token‑aware chunk selection under a configurable budget with a focused answer prompt.

Personal Memory
- Advanced pipeline uses the LLM to:
  - classify intent (memory vs rag vs chitchat),
  - extract personal facts (name, likes, allergies, goals, preferences, etc.),
  - answer personal questions using only stored memory.
- Memory is keyed by `user_id`. New conversations reuse the same user memory.

Quickstart
1) Prerequisites
- Python 3.10+
- OpenAI API Key

2) Setup
- Create a `.env` (see `.env.example`) with your keys:
  - `OPENAI_API_KEY="sk-..."`
  - Optional: `COHERE_API_KEY` for reranking, Postgres DSN for pgvector, etc.

3) Install
```
pip install -r backend/requirements.txt
```

4) Run the API
```
uvicorn backend.app.main:app --reload
```
By default the app exposes routes under `/api` (see below).

5) Ingest Data (optional but recommended)
- Add files under `backend/app/data` (txt/md/mdx/pdf/docx) or ingest via API:
```
POST /api/ingest { "content": "your text", "title": "Optional Title" }
POST /api/ingest_url { "url": "https://...", "title": "Optional Title" }
```

API Endpoints (selected)
- Health: `GET /api/health`
- Ask: `POST /api/ask`
  - body: `{ user_id?: string, session_id?: string, question: string, top_k?: number }`
  - returns: `{ answer: string, context: RetrievedChunk[] }`
- Ingest text: `POST /api/ingest` → `{ chunks_indexed: number }`
- Ingest URL: `POST /api/ingest_url` → `{ chunks_indexed: number }`
- Sessions
  - `POST /api/sessions/create` → `{ session_id }`
  - `GET /api/sessions/list?user_id=...` → list of sessions
  - `GET /api/sessions/{session_id}/history` → summary + messages
  - `POST /api/sessions/{session_id}/reset` → reset messages/summary
  - `DELETE /api/sessions/{session_id}` → delete session
- Simple conversation history (in‑process memory)
  - `GET /api/conversation/{user_id}/history`
  - `DELETE /api/conversation/{user_id}`

Configuration (env)
- OpenAI
  - `OPENAI_API_KEY` (required)
  - `OPENAI_CHAT_MODEL` (default: `gpt-3.5-turbo`)
  - `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
  - `OPENAI_TEMPERATURE` (default: `0.3`)
- RAG
  - `RAG_TOP_K` (default: `4`)
  - `ENABLE_HYBRID` (default: `true`), `HYBRID_ALPHA` (default: `0.5`)
  - `ENABLE_RERANK` (default: `true`), `RERANK_PROVIDER` (`cohere|local|none`), `RERANK_TOP_N`
  - `VECTOR_BACKEND` (`faiss|pgvector`), `POSTGRES_DSN` for pgvector
  - Chunking: `CHUNKER` (`token|char`), `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `MAX_INPUT_TOKENS`
- Memory
  - `ENABLE_MEMORY` (default: `true`)
  - `MEMORY_BACKEND` (`sqlite|postgres`), `MEMORY_DSN` (default: `sqlite:///backend/memory.db`)
  - Summaries: `HISTORY_WINDOW`, `SUMMARY_EVERY_N_TURNS`, `SUMMARY_MAX_TOKENS`

Development
- Run tests
```
pytest
```

- Lint/format (suggested)
```
ruff check .
black .
```

Security & Privacy Notes
- User memory stores personal information. Ensure you:
  - secure your DB and vector store directories,
  - implement auth on API routes,
  - provide a “forget me” flow to delete user data,
  - avoid logging PII.
- FAISS loading uses `allow_dangerous_deserialization=True` for local persistence. Only load from trusted directories.

Roadmap Ideas
- Add an authenticated “Forget Me” endpoint to purge user memory and sessions.
- Add per‑(user, session) memory scoping in the simple pipeline.
- Add observability (structured logs, latency and usage metrics).
- Support additional rerankers and safety‑tuned prompts.

License
This project is for demonstration and educational purposes. Add a license file before distributing.

