"""
Local Hugging Face embedding model factory.

Uses a sentence-transformer model locally, so PDF ingestion
does not require paid OpenAI embedding API calls.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from app.utils.config import settings


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
