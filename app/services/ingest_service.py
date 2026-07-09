"""
Orchestrates the ingestion pipeline:
save upload -> load -> split -> embed -> store.

Supports both single-file and multi-file PDF ingestion.
"""
import os
import shutil
from typing import List

from fastapi import UploadFile

from app.loaders.pdf_loader import load_and_split
from app.vectorstore.chroma_store import add_documents, get_collection_count
from app.utils.config import settings


def save_upload(file: UploadFile) -> str:
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    filename = os.path.basename(file.filename)
    dest_path = os.path.join(settings.UPLOAD_DIR, filename)

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return dest_path


def ingest_pdf(file: UploadFile) -> dict:
    file_path = save_upload(file)
    chunks = load_and_split(file_path)
    ids = add_documents(chunks)

    return {
        "filename": file.filename,
        "chunks_indexed": len(ids),
        "total_chunks_in_store": get_collection_count(),
    }


def ingest_pdfs(files: List[UploadFile]) -> dict:
    documents = []
    total_new_chunks = 0

    for file in files:
        file_path = save_upload(file)
        chunks = load_and_split(file_path)
        ids = add_documents(chunks)

        chunk_count = len(ids)
        total_new_chunks += chunk_count

        documents.append(
            {
                "filename": file.filename,
                "chunks_indexed": chunk_count,
            }
        )

    return {
        "files_processed": len(documents),
        "documents": documents,
        "total_new_chunks": total_new_chunks,
        "total_chunks_in_store": get_collection_count(),
    }
