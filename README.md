# AI Nutrition Assistant

An agentic, Retrieval‑Augmented Generation (RAG) nutrition coach that remembers you, searches your knowledge base, and chats naturally. Built with LangGraph + LangChain, FastAPI, and Streamlit, with persistent per‑user memory and session management.

Product overview
- Agentic RAG app: A small “brain” routes each message to Memory, RAG, or Chit‑chat using an LLM classifier (LangGraph state machine).
- Personal: Learns name, preferences, goals, allergies, and uses them to personalize advice.
- Grounded: Answers with context retrieved from your private corpus (documents, PDFs, URLs).
- Manageable: Sessions and user memory persist across conversations; you can reset or delete them.

Highlights
- Personalized coaching: “Build me a high‑protein vegetarian plan for my goals.”
- Memory queries: “What did I say I’m allergic to?”
- Trustworthy answers: “Compare quinoa vs rice for my macros” with sourced context.
- Simple ops: One FastAPI app + Streamlit UI, env‑driven config.

Tech stack
- Orchestration: LangGraph (agentic routing/graph), LangChain (chains, prompts, memory)
- LLMs: OpenAI Chat (responses, intent), OpenAI Embeddings (dense vectors)
- Retrieval: Hybrid RAG = FAISS dense + BM25 keyword; optional Cohere reranker or local reranker
- Backend: FastAPI (REST API), SQLAlchemy (SQLite/Postgres) for conversations + user key‑value memory
- Frontend: Streamlit (starter UI) in `frontend/`
- Utilities: tiktoken, rank_bm25, pypdf/docx/trafilatura loaders (optional)

How it works
- Agentic flow (LangGraph)
  - Classifier node: LLM labels message as memory | rag | chitchat.
  - Memory node: LLM extracts personal facts (per user_id) or answers using only stored facts.
  - RAG node: Retrieves via FAISS + BM25, optionally reranks, builds a token‑aware context, then answers.
  - Chit‑chat node: Lightweight conversational fallback.
- Memory model
  - Per‑user key/value memory stored in SQL (`user_id` scoped).
  - Conversation history + rolling summaries per session (create/list/history/reset/delete).
  - Simple pipeline also supports in‑process LangChain memory (buffer/summary) for quick prototypes.

Key components
- API routes: `backend/app/api/routes.py`
- Agentic pipeline: `backend/app/services/agentic_pipeline.py`
- Simple pipeline: `backend/app/services/simple_agentic_pipeline.py`
- RAG service: `backend/app/services/rag_service.py`
- Memory service: `backend/app/services/memory_service.py`
- Settings: `backend/app/config.py`

Setup
- Prereqs: Python 3.10+, OpenAI API key
- Configure: copy `.env.example` → `.env` and set `OPENAI_API_KEY` (and optional `COHERE_API_KEY`, Postgres DSN if using pgvector)
- Install: `pip install -r backend/requirements.txt`
- Run API: `uvicorn backend.app.main:app --reload`
- Optional UI: run Streamlit app in `frontend/` (e.g., `streamlit run frontend/app.py`)

Using the API
- Ask: `POST /api/ask` with `{ user_id, question, session_id? }`
- Ingest text: `POST /api/ingest`
- Ingest URL: `POST /api/ingest_url`
- Sessions: create/list/history/reset/delete under `/api/sessions`

Configuration (selected)
- RAG: `ENABLE_HYBRID`, `HYBRID_ALPHA`, `ENABLE_RERANK`, `RERANK_PROVIDER`, `RAG_TOP_K`, `VECTOR_BACKEND`
- Memory: `ENABLE_MEMORY`, `MEMORY_DSN`, `HISTORY_WINDOW`, `SUMMARY_EVERY_N_TURNS`
- Models: `OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_TEMPERATURE`

Privacy & control
- Memory is per user_id and scoped to your database. Add auth in front of the API and expose a “Forget Me” action (delete user memory + sessions) for production.

Roadmap
- “Forget Me” endpoint, richer citations, multi‑turn tool use.

PRs and feedback welcome.
