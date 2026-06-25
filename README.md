# PDF RAG Assistant — AIO2026 Project 1.2 (OpenAI Edition)

A Streamlit chatbot that lets you upload any PDF and ask questions against it,
powered by the OpenAI API (no local GPU required).

## Architecture

```
PDF upload → pypdf (text) → chunking → text-embedding-3-small → ChromaDB
User query → embed query → vector search → top-4 chunks → gpt-4o-mini → answer
```

## Models used

| Role      | Model                    | Why                                  |
|-----------|--------------------------|--------------------------------------|
| Embedding | `text-embedding-3-small` | Fast, cheap, 1536-dim, multilingual  |
| Generation| `gpt-4o-mini`            | Low latency, low cost, high quality  |
| Vector DB | ChromaDB (in-memory)     | Zero infra, per-session              |

## Quick start

### 1. Clone / download the files

```bash
git clone 
```

### 2. Install dependencies

```bash (Mac)
pip install -r requirements.txt
```

```bash (Windows)
python -m pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash (Mac)
cp .env.example .env
# edit .env and paste your key

# or export directly:
export OPENAI_API_KEY=sk-...
```

```bash (Windows)
copy .env.example .env
# edit .env and paste your key

# or export directly:
set OPENAI_API_KEY=sk-...
```

### 4. Run

```bash (Mac)
streamlit run chatbot_app.py
```

```bash (Windows)
python -m streamlit run chatbot_app.py
```

Open http://localhost:8501 in your browser.

## Usage

1. **Upload PDF** — sidebar → "Choose a PDF file"
2. **Process** — click "⚙️ Process PDF" (indexes the document)
3. **Chat** — type your question in the chat box
4. **Clear** — "🗑️ Clear chat history" to reset

## Configuration (top of chatbot_app.py)

| Constant     | Default                    | Description                       |
|--------------|----------------------------|-----------------------------------|
| `EMBED_MODEL`| `text-embedding-3-small`   | OpenAI embedding model            |
| `CHAT_MODEL` | `gpt-4o-mini`              | OpenAI chat model                 |
| `CHUNK_SIZE` | `1000`                     | Characters per chunk              |
| `OVERLAP`    | `200`                      | Overlap between adjacent chunks   |
| `TOP_K`      | `4`                        | Chunks retrieved per query        |

## Differences from the AIO2026 Ollama reference

| Component  | Original (Ollama)          | This version (OpenAI)             |
|------------|----------------------------|-----------------------------------|
| Embeddings | `bge-m3` via Ollama        | `text-embedding-3-small` via API  |
| LLM        | `vicuna:7b-v1.5-q5_1`     | `gpt-4o-mini` via API             |
| Setup      | Download ~6 GB models      | Just an API key                   |
| GPU        | Required for good speed    | Not needed                        |
| Cost       | Free after download        | ~$0.001 per typical question      |

## Cost estimate

- Embedding a 50-page PDF ≈ **$0.001**
- Each question (embed + 4 chunks + answer) ≈ **$0.001–0.003**
- 100 questions on a document ≈ **< $0.30**
