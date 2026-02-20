# 🍳 Recipe RAG CLI (v1)

A lightweight Retrieval-Augmented Generation (RAG) system for answering questions about recipes using embeddings, ChromaDB, and an LLM — all from the command line.

---

## 📌 Overview

This project implements a minimal, reproducible RAG pipeline:

- 📄 Markdown recipes as source data  
- ✂️ Chunking + JSONL storage  
- 🧠 Embeddings stored in Chroma  
- 🔎 Top-k semantic retrieval  
- 🤖 LLM generation with sources  
- 💻 Simple CLI interface  

---

# 🏗 Architecture

## Data Flow Diagram

### 🧱 Ingest / Index (One-Time or Occasional)

```
recipes.md
    ↓
chunks.jsonl
    ↓ (embed)
Chroma DB (persist/recipes_v1)
```

**Process:**

1. Parse `recipes.md`
2. Split into chunks
3. Save chunks to `chunks.jsonl`
4. Generate embeddings
5. Store vectors in Chroma (persistent)

---

### 💬 Run (Per Question)

```
CLI question
    ↓ (embed query)
Chroma top-k
    ↓ (select best-1)
LLM
    ↓
Answer + Sources footer
```

**Process:**

1. User asks a question via CLI  
2. Embed the query  
3. Retrieve top-k relevant chunks  
4. Select best match  
5. Send context to LLM  
6. Return answer with source attribution  

---

# 📁 Project Structure (v1)

```
.
├── data/
│   ├── recipes.md
│   ├── chunks.jsonl
│   └── chroma/                # Chroma persist directory
│
├── src/
│   ├── parse/                 # Optional (for reproducibility)
│   ├── index/
│   │   └── build_index.py
│   ├── rag/
│   │   ├── retrieve.py
│   │   ├── prompt.py
│   │   └── generate.py
│   └── cli.py
│
├── configs/
│   └── v1.yaml                # paths, embedding model, k, LLM model
│
├── logs/
│   └── queries.jsonl
│
└── docs/
    └── architecture_v1.md     # Add diagram image later
```

---

# ⚙️ Configuration

Example `configs/v1.yaml`:

```yaml
paths:
  recipes: data/recipes.md
  chunks: data/chunks.jsonl
  chroma: data/chroma/

embedding:
  model: nomic-embed-text
  k: 5

llm:
  model: llama3
```

---

# 🚀 Usage

### 1️⃣ Build the Index

```bash
python src/index/build_index.py
```

### 2️⃣ Ask a Question

```bash
python src/cli.py "How do I make fluffy pancakes?"
```

---

# 🧠 System Components

| Layer        | Responsibility |
|-------------|----------------|
| Data Layer  | Markdown → JSONL chunks |
| Vector Layer | Embeddings stored in Chroma |
| Retrieval Layer | Top-k semantic search |
| Generation Layer | Prompt + context → LLM |
| Interface | CLI |

---

# 📝 Logging

All queries are logged for evaluation and debugging:

```
logs/queries.jsonl
```

Each entry may include:

- User question  
- Retrieved chunks  
- Selected context  
- Final answer  

---

# 🔮 Future Improvements

- Evaluation pipeline  
- Reranking step  
- Streaming responses  
- Web UI  
- Structured source citations  
- Multi-document support  

---

# 🏷 Version

**v1 — Minimal, reproducible, CLI-based RAG system**

---

Built for clarity, iteration, and extensibility.