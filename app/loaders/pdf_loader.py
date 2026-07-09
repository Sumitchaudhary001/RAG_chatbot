"""
Document ingestion: load PDFs and split them into chunks.

Design choices (explained in README.md Part 1):
- PyPDFLoader: simple, dependency-light, page-aware (keeps page_number
  metadata per chunk, which we need for citations later).
- RecursiveCharacterTextSplitter: tries to split on paragraph -> sentence
  -> word boundaries in that order, so chunks stay semantically coherent
  instead of being cut mid-sentence like a fixed-width splitter would do.
"""
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.utils.config import settings


def load_pdf(file_path: str) -> List[Document]:
    """Load a single PDF into one LangChain Document per page."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Tag every page with a clean source filename for later citation.
    filename = os.path.basename(file_path)
    for page in pages:
        page.metadata["source"] = filename

    return pages


def split_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """Split loaded documents into retrieval-sized chunks."""
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Give each chunk a stable, human-readable id: filename + page + index.
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        chunk.metadata["chunk_id"] = f"{source}-p{page}-c{i}"

    return chunks


def load_and_split(file_path: str) -> List[Document]:
    """Convenience wrapper used by the ingestion service."""
    pages = load_pdf(file_path)
    return split_documents(pages)
