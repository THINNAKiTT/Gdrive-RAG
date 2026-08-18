# Gdrive-RAG

A RAG chatbot that answers questions
about documents stored in a Google Drive folder. Supports both fully local
operation (Ollama + local reranker, no API keys, no data leaving your machine)
and cloud providers (OpenAI, Gemini, Claude, Voyage AI, Cohere) -- mix and
match per component.

## Features

- **Auto-sync from Google Drive** -- a background daemon watches your Drive
  folder via the Changes API and incrementally re-indexes new/modified/deleted
  files, without blocking the chat UI.
- **Multi-format ingestion** -- PDF, EPUB, plain text, and images (via OCR).
- **Session memory** -- SQLite-backed chat history with pin/rename/delete,
  isolated per session, plus a small local model that resolves conversational
  context ("How does *it* work?") before retrieval.
- **Two-stage retrieval** -- vector search casts a wide net, then a
  cross-encoder reranker (local BGE-M3 or Cohere) picks the best matches.
- **Provider-agnostic** -- swap the main LLM, embedding model, query rewriter,
  and reranker independently between local (Ollama) and cloud (OpenAI, Gemini,
  Claude, Voyage, Cohere) via a guided setup wizard -- no manual dependency
  wrangling.
- **Resilient** -- retry with exponential backoff and a circuit breaker around
  every AI provider call, so a flaky connection degrades gracefully instead of
  crashing the sync daemon or the chat.
- **Structured (JSON) logging** for every component.
- **ChromaDB viewer** -- inspect what's actually indexed via a CSV export or a
  standalone web UI (table + t-SNE embedding graph).
- **Dockerized** -- one image, three services (chat app, sync daemon, viewer)
  via docker-compose.

## Architecture

```
Google Drive (Changes API)
|
v
sync_daemon.py --(file lock)--> ChromaDB (./chroma_db)
^
|
ui/app.py (Streamlit) <---- query -----+
|
+--> QueryRewriter (resolves conversational context)
+--> Vector search (top 15) --> Reranker (top 4) --> LLM --> answer
+--> SQLite (chat_history.db) -- session memory
```

Every AI-backed component (main LLM, embedding model, query rewriter,
reranker) is resolved at runtime through `src/rag/providers.py`, based on
`*_PROVIDER` values in `.env`. Local providers need no API key; cloud
providers need the matching key set in `.env`.

## Quick start

**Recommended package manager: [uv](https://docs.astral.sh/uv/).**

### 1. Clone and set up Google Drive credentials

```bash
git clone https://github.com/THINNAKiTT/Gdrive-RAG.git
cd Gdrive-RAG
```

Place a Google Cloud service account JSON key at
`config/secure_gcp_credentials.json`, and share your target Drive folder with
that service account's email. Copy `.env.example` to `.env` and set
`GOOGLE_DRIVE_FOLDER_ID`.

### 2. Run the app

```bash
uv run python run.py
```

That's it. `run.py` checks whether the providers selected in `.env` have
their SDKs installed; if not (e.g. first run, or you just switched a
provider), it launches an interactive setup wizard that asks which provider
to use for each of the four roles below, writes your choice into `.env`, and
runs `uv sync` with exactly the extras you need -- then starts the app.

| Role | What it does | Providers |
|---|---|---|
| `LLM_PROVIDER` | Answers questions | `local` (Ollama), `openai`, `gemini`, `claude` |
| `EMBEDDING_PROVIDER` | Indexes documents & queries | `local` (Ollama), `openai`, `gemini`, `voyage` |
| `QUERY_REWRITE_PROVIDER` | Resolves conversational context | `local` (Ollama), `openai`, `gemini`, `claude` |
| `RERANKER_PROVIDER` | Re-scores retrieved chunks | `local` (BGE-M3), `cohere` |

Cloud providers need their API key set in `.env` (`OPENAI_API_KEY`,
`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `COHERE_API_KEY`).
`local` needs [Ollama](https://ollama.com) running, with the models pulled:

```bash
ollama pull llama3
ollama pull nomic-embed-text
ollama pull qwen2.5:0.5b
```

The reranker's `local` provider needs no external service, but downloads and
caches the `BAAI/bge-reranker-v2-m3` model (~2.3GB) on first use.

**Re-running setup manually** (e.g. to change providers later):

```bash
uv run python src/setup_wizard.py
```

**Non-interactive setup** (for scripting/CI -- reads `*_PROVIDER` values
already in `.env`, no prompts):

```bash
uv run python src/setup_wizard.py --non-interactive
```

### 3. Start the sync daemon (separate terminal)

The chat app reads from ChromaDB but doesn't sync Drive itself -- that's the
daemon's job, kept separate so a slow sync never blocks a chat response.

```bash
uv run python -m src.sync_daemon
```

It does a full sync on first run, then polls Drive's Changes API every 30
seconds (configurable via `POLL_INTERVAL_SECONDS` in `.env`) for incremental
updates.

### 4. (Optional) Inspect the vector store

```bash
# CSV export
uv run python chroma_viewer.py --export-csv export.csv

# Interactive web viewer (table + t-SNE embedding graph), on a separate port
uv run python chroma_viewer.py --web
```

## Manual installation (without `run.py`)

If you'd rather manage dependencies yourself:

```bash
# Local only (Ollama + local reranker):
uv sync --extra local

# A specific cloud provider:
uv sync --extra openai
uv sync --extra gemini
uv sync --extra claude
uv sync --extra voyage
uv sync --extra cohere

# Everything:
uv sync --extra all

# Multiple extras must be combined in ONE command -- `uv sync --extra X`
# followed by `uv sync --extra Y` UNINSTALLS X's packages, it doesn't add
# to them:
uv sync --extra openai --extra reranker-local
```

Then run components directly:

```bash
uv run streamlit run ui/app.py
uv run python -m src.sync_daemon
uv run python chroma_viewer.py --web
```

## Docker

One image, three services, sharing `./chroma_db`, `./logs`, and a
`huggingface_cache` volume (so the reranker model downloads once and
persists across restarts) with standalone ChromaDB.

```bash
# .env must already be configured (run.py / setup_wizard.py on the host
# first -- .env is mounted read-only into containers, so the wizard can't
# write to it from inside one)
docker-compose build
docker-compose up
```

- Chat app: http://localhost:8501
- Standalone chroma db server on port `8000`
- Sync daemon runs headless (check `docker-compose logs -f sync_daemon`)

The image is built with `--extra all` (every provider's SDK included), so
you can switch providers via `.env` without rebuilding.

### (Optional) Inspect the vector store
```
# CSV export
uv run python chroma_viewer.py --export-csv export.csv

# Interactive web viewer (table + t-SNE embedding graph), on a separate port
uv run python chroma_viewer.py --web
```

## Configuration reference

See `.env.example` for the full list. Key groups:

- **Google Drive**: `GCP_CREDENTIALS_PATH`, `GOOGLE_DRIVE_FOLDER_ID`
- **Provider selection**: `LLM_PROVIDER`, `EMBEDDING_PROVIDER`,
  `QUERY_REWRITE_PROVIDER`, `RERANKER_PROVIDER`
- **Local (Ollama)**: `OLLAMA_URL`, `OLLAMA_MODEL`, `EMBEDDING_MODEL`,
  `QUERY_REWRITE_MODEL`, `RERANKER_MODEL`
- **Cloud API keys & models**: `OPENAI_API_KEY` + `OPENAI_LLM_MODEL` /
  `OPENAI_EMBEDDING_MODEL` / `OPENAI_QUERY_REWRITE_MODEL`,
  `GEMINI_API_KEY` + matching model vars, `ANTHROPIC_API_KEY` + matching
  model vars, `VOYAGE_API_KEY`, `COHERE_API_KEY`

## Testing

```bash
# Fast, fully-isolated tests -- no network, no external services:
uv run pytest -m unit -v

# Integration tests -- require a real .env, running Ollama, and a live
# Drive folder:
uv run pytest -m integration -v
```

## Project structure
```
src/
ingestion/ Drive client, multi-format document parsing (PDF/EPUB/text/OCR)
storage/ ChromaDB manager, incremental sync engine, SQLite chat history, sync lock
rag/ Orchestrator, provider factory, query rewriter, reranker, prompts
utils/ Structured logging, retry/circuit breaker
sync_daemon.py Standalone background sync process
setup_wizard.py Interactive/non-interactive provider setup
ui/app.py Streamlit chat interface
chroma_viewer.py Standalone ChromaDB inspection tool (CSV export + web UI)
run.py Entry point: pre-flight checks providers, launches setup if needed
tests/
unit/ Fast, isolated (mocked) tests
integration/ Tests requiring real Drive/Ollama credentials
```