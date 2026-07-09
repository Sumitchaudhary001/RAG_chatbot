"""
Centralized configuration loaded from environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- LLM / Embeddings ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    CHAT_MODEL: str = os.getenv(
        "CHAT_MODEL",
        "llama-3.1-8b-instant"
    )
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Chunking ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 150))

    # --- Retrieval ---
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", 4))

    # --- Memory ---
    MAX_HISTORY_MESSAGES: int = int(
        os.getenv("MAX_HISTORY_MESSAGES", 10)
    )

    # --- Storage ---
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR",
        "./data/chroma_db"
    )
    UPLOAD_DIR: str = os.getenv(
        "UPLOAD_DIR",
        "./data/uploads"
    )
    COLLECTION_NAME: str = os.getenv(
        "COLLECTION_NAME",
        "rag_documents"
    )


settings = Settings()


if not settings.GROQ_API_KEY:
    print(
        "[WARNING] GROQ_API_KEY is not set. "
        "Add your Groq API key to the .env file."
    )
