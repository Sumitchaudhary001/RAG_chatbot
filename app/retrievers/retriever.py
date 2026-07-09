"""
Retriever construction.

A Retriever wraps a vector store with a search strategy (similarity,
mmr, similarity_score_threshold, etc). Keeping it separate from the
raw vector store means we can change search strategy or add
reranking/hybrid search later without touching the chain code.
"""
from langchain_core.vectorstores import VectorStoreRetriever

from app.vectorstore.chroma_store import get_vectorstore
from app.utils.config import settings


def get_retriever(k: int = None) -> VectorStoreRetriever:
    k = k or settings.RETRIEVAL_K
    store = get_vectorstore()
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
