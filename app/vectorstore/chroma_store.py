"""
ChromaDB vector store wrapper.

Handles creating/loading a persistent Chroma collection and adding
new document chunks to it. Persistence means we embed each document
only once -- restarting the server does not require re-ingesting.
"""
import os
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.embeddings.embedder import get_embedding_model
from app.utils.config import settings

_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    """Return a singleton Chroma instance backed by a persistent directory."""
    global _vectorstore
    if _vectorstore is None:
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
        _vectorstore = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=get_embedding_model(),
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
    return _vectorstore


def add_documents(chunks: List[Document]) -> List[str]:
    """Embed and store chunks. Returns the ids Chroma assigned them."""
    store = get_vectorstore()
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    store.add_documents(documents=chunks, ids=ids)
    return ids


def get_collection_count() -> int:
    store = get_vectorstore()
    return store._collection.count()
