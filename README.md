# Cooking Assistant (Moroccan Recipe RAG)

A retrieval-augmented cooking assistant that answers recipe questions from a local Moroccan recipes dataset.

It includes:
- Hybrid retrieval (dense embeddings + BM25)
- Query routing for recipe-focused search
- LLM generation constrained to retrieved context
- FastAPI backend
- Streamlit frontend
- CLI and retrieval evaluation scripts

## Project Architecture

Query flow:
1. User query (CLI / API / Streamlit)
2. Query router rewrites the query to a recipe-search form
3. Hybrid retriever fetches top-k recipe chunks from ChromaDB
4. Generator prompts an Ollama model with retrieved context
5. Response returned with source recipe names

Core modules:
- `src/rag/route_query.py`: query normalization/routing with `qwen2.5:1.5b`
- `src/rag/retrieve.py`: Chroma + BM25 ensemble retrieval
- `src/rag/generate.py`: answer generation with `qwen3:4b`
- `src/rag/orchestrate.py`: end-to-end orchestration

## Repository Structure

```text
src/
  backend/main.py         FastAPI app (`/health`, `/generate/{query_text}`)
  frontend/streamlit.py   Simple Streamlit UI
  rag/                    Router, retriever, generator, orchestrator
  index/indexer.py        Builds/rebuilds Chroma index from JSON chunks
  eval/evaluate.py        Hit@k / MRR@k evaluation
  cli.py                  Command-line query interface

data/
  recipe_chunks.json      Recipe chunk source for indexing
  chroma/                 Persisted vector store
  original_set/           Original CSV dataset
```

## Requirements

- Python `>=3.12,<3.14`
- [Poetry](https://python-poetry.org/)
- [Ollama](https://ollama.com/) running locally
- Ollama models used by default:
  - `qwen2.5:1.5b` (query router)
  - `qwen3:4b` (answer generation)

Optional:
- CUDA-enabled GPU for faster embedding/model workloads

## Setup

```bash
# 1) Install dependencies
poetry install

# 2) Make sure Ollama is running, then pull models
ollama pull qwen2.5:1.5b
ollama pull qwen3:4b
```

## Build or Rebuild the Vector Index

Run when `data/recipe_chunks.json` changes:

```bash
poetry run python -m src.index.indexer
```

This writes/updates the Chroma database under `data/chroma/`.

## Run the App

### 1) Start backend API

```bash
poetry run uvicorn src.backend.main:app --reload
```

- Health check: `GET http://127.0.0.1:8000/health`
- Generate answer: `GET http://127.0.0.1:8000/generate/{query_text}`

### 2) Start Streamlit frontend

In a second terminal:

```bash
poetry run streamlit run src/frontend/streamlit.py
```

The frontend calls the backend at `http://127.0.0.1:8000`.

## CLI Usage

```bash
poetry run python -m src.cli "how do i make chicken tagine"
```

The CLI prints:
- Generated answer
- Retrieved source recipe name(s)

## Evaluation

Run retrieval evaluation (Hit@k and MRR@k):

```bash
poetry run python -m src.eval.evaluate
```

Evaluation queries are in:
- `src/eval/evaluation_queries_50_paraphrased.json`

## Notes and Current Limitations

- Some default paths in `src/rag/retrieve.py` and `src/eval/evaluate.py` are absolute Windows paths. For portability, update them to relative paths or pass paths explicitly.
- `src/frontend/streamlit.py` currently sends a request whenever the input changes; adding a submit button can reduce unnecessary requests.
- The backend endpoint uses path parameters for full query text. URL encoding is required for special characters.

## Tech Stack

- FastAPI, Streamlit
- LangChain + Ollama
- ChromaDB
- HuggingFace BGE embeddings (`BAAI/bge-small-en-v1.5`)
- BM25 + ensemble retrieval
- Poetry for dependency management
