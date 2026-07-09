# RAG Chatbot using Langchain

A full-stack Retrieval-Augmented Generation (RAG) application for asking grounded questions over one or multiple PDF documents.

The system provides:
- Multi-PDF document ingestion
- Recursive text chunking with overlap
- Local Hugging Face embeddings
- Persistent ChromaDB vector storage
- Semantic document retrieval
- Groq-hosted Llama generation
- Source filename, page number, and evidence snippets
- Session-based conversational memory
- FastAPI backend
- Streamlit frontend
- 20-question evaluation harness

## Architecture

```text
PDF Documents
      |
      v
PyPDFLoader
      |
      v
RecursiveCharacterTextSplitter
      |
      v
Hugging Face Embeddings
sentence-transformers/all-MiniLM-L6-v2
      |
      v
Persistent ChromaDB
      |
      v
Retriever (Top-K = 6)
      |
      v
Relevant Context
      |
      v
Grounded RAG Prompt
      |
      v
Groq-hosted Llama Model
      |
      v
Answer + Source Evidence
```

## Tech Stack

- Python 3.11
- LangChain
- FastAPI
- Streamlit
- Groq
- Llama 3.1 8B Instant
- Hugging Face Sentence Transformers
- ChromaDB
- PyPDFLoader
- Pydantic

## Project Structure

```text
rag-chatbot/
├── app/
│   ├── api/              # FastAPI routes
│   ├── chains/           # LCEL RAG chain and memory
│   ├── embeddings/       # Hugging Face embedding factory
│   ├── loaders/          # PDF loading and chunking
│   ├── models/           # Pydantic schemas
│   ├── prompts/          # Grounded RAG prompts
│   ├── retrievers/       # Retriever configuration
│   ├── services/         # Ingestion and chat orchestration
│   ├── utils/            # Environment configuration
│   ├── vectorstore/      # Persistent ChromaDB wrapper
│   └── main.py           # FastAPI entry point
├── eval/
│   ├── test_questions.json
│   └── evaluate.py
├── frontend/
│   └── streamlit_app.py
├── .env.example
├── requirements.txt
└── README.md
```

## Features

### Multi-Document PDF Ingestion

Users can upload one or multiple PDF documents. Each PDF is:

1. Loaded with `PyPDFLoader`
2. Split into overlapping chunks
3. Embedded locally using Hugging Face embeddings
4. Stored persistently in ChromaDB
5. Tagged with source filename and page metadata

Current chunking configuration:

```text
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

### Retrieval-Augmented Generation

For every question:

1. The retriever searches ChromaDB
2. The top 6 relevant chunks are selected
3. Retrieved context is formatted for the prompt
4. The Groq-hosted Llama model generates an answer
5. Source evidence is returned with filename, page, and snippet

### Grounded Answering

The prompt instructs the model to:

- Answer from retrieved context
- Avoid unsupported claims
- Keep responses concise
- Cite sources where possible
- Respond with "I don't know based on the provided documents." when the answer is unsupported

### Conversational Memory

Conversation history is tracked using a client-provided `session_id`.

The system uses:

- `RunnableWithMessageHistory`
- `InMemoryChatMessageHistory`
- bounded history trimming

This allows follow-up questions while preventing unlimited context growth.

### Source Transparency

Each answer can expose retrieved evidence including:

- Source filename
- Page number
- Text snippet

### Streamlit Frontend

The single-page frontend supports:

- Multiple PDF uploads
- Document indexing
- Question answering
- Generated answers
- Retrieved source evidence
- Question history
- Conversation reset

## Setup

### 1. Create a Python 3.11 virtual environment

```bash
python3.11 -m venv .venv
```

Activate it:

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example file:

```bash
cp .env.example .env
```

Add your Groq API key to `.env`:

```env
GROQ_API_KEY=your_actual_groq_api_key
CHAT_MODEL=llama-3.1-8b-instant

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

CHUNK_SIZE=500
CHUNK_OVERLAP=100
RETRIEVAL_K=6

MAX_HISTORY_MESSAGES=10

CHROMA_PERSIST_DIR=./data/chroma_db
UPLOAD_DIR=./data/uploads
COLLECTION_NAME=rag_documents
```

Never commit the real `.env` file.

## Running the Application

The backend and frontend run as separate processes.

### Terminal 1 — FastAPI Backend

```bash
cd ~/Desktop/rag-chatbot
source .venv/bin/activate
uvicorn app.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

### Terminal 2 — Streamlit Frontend

```bash
cd ~/Desktop/rag-chatbot
source .venv/bin/activate
streamlit run frontend/streamlit_app.py
```

Frontend:

```text
http://localhost:8501
```

## API Endpoints

### Health Check

```text
GET /health
```

### Ingest One or Multiple PDFs

```text
POST /ingest
```

The endpoint accepts multiple files using the multipart field name `files`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

Example response:

```json
{
  "files_processed": 2,
  "documents": [
    {
      "filename": "document1.pdf",
      "chunks_indexed": 12
    },
    {
      "filename": "document2.pdf",
      "chunks_indexed": 18
    }
  ],
  "total_new_chunks": 30,
  "total_chunks_in_store": 30
}
```

### Chat

```text
POST /chat
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-1",
    "question": "What is this document about?"
  }'
```

Example response:

```json
{
  "answer": "The document describes...",
  "sources": [
    {
      "source": "document.pdf",
      "page": 1,
      "snippet": "Relevant retrieved text..."
    }
  ],
  "session_id": "demo-1"
}
```

Reuse the same `session_id` for follow-up questions in the same conversation.

## Evaluation

The project includes a 20-question evaluation harness.

Each evaluation record contains:

- Retrieved documents
- Generated answer
- Expected answer
- Automatic overlap score
- Correct/incorrect heuristic result

Run:

```bash
python -m eval.evaluate
```

Results are written to:

```text
eval/eval_results.csv
eval/eval_results.json
```

### Retrieval Tuning Experiment

An initial evaluation exposed retrieval failures for several questions.

Baseline configuration:

```text
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
RETRIEVAL_K=4
```

Baseline automatic heuristic result:

```text
10/20 passes
```

The retrieval configuration was then tuned to:

```text
CHUNK_SIZE=500
CHUNK_OVERLAP=100
RETRIEVAL_K=6
```

After tuning:

```text
13/20 automatic heuristic passes
```

This improved the automatic evaluation result from:

```text
50% -> 65%
```

Manual review indicated that several additional answers were substantively correct but marked false by the lexical word-overlap heuristic because the generated wording differed from the expected answer.

Therefore, automatic overlap scores are treated as a first-pass signal and final correctness should be manually reviewed.

## Design Decisions

### Why RecursiveCharacterTextSplitter?

It attempts to preserve:

1. Paragraph boundaries
2. Line boundaries
3. Sentence boundaries
4. Word boundaries

This produces more coherent retrieval chunks than fixed-width splitting.

### Why Chunk Overlap?

Overlap helps preserve information that crosses chunk boundaries. Without overlap, a relevant sentence or concept may be split between chunks and become harder to retrieve.

### Why Hugging Face Embeddings?

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Advantages:

- Runs locally
- No embedding API cost
- Suitable for semantic similarity search
- Avoids dependency on a paid embedding provider

### Why ChromaDB?

ChromaDB provides:

- Vector similarity search
- Persistent local storage
- LangChain integration
- Simple development setup

### Why Groq?

Groq provides fast hosted inference for the chat model. The current model is:

```text
llama-3.1-8b-instant
```

### Why FastAPI?

FastAPI provides:

- Typed request/response validation
- Automatic Swagger documentation
- Clean separation between API and RAG logic
- Easy frontend integration

### Why Streamlit?

Streamlit provides a lightweight interface for:

- Multi-document uploads
- Question answering
- Source inspection
- Local project demonstration

## Limitations

Current limitations include:

- PDF-only ingestion
- In-memory conversational history is lost when the backend restarts
- Dense semantic retrieval can miss exact identifier-based queries
- The automatic evaluation heuristic is lexical and can mark semantically correct answers as incorrect
- Local ChromaDB storage is intended primarily for development and demonstration

## Future Improvements

Potential improvements include:

- Hybrid dense + keyword retrieval
- Reranking
- Metadata filtering
- Persistent Redis/PostgreSQL conversation history
- Stronger semantic evaluation
- Authentication
- Cloud deployment
- Additional document formats

## Security

- `.env` is excluded from Git
- API keys must never be committed
- Uploaded documents are excluded from Git
- Local ChromaDB data is excluded from Git
