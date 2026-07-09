from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    MultiIngestResponse,
)
from app.services.ingest_service import ingest_pdfs
from app.services.chat_service import handle_chat

router = APIRouter()


@router.post("/ingest", response_model=MultiIngestResponse)
async def ingest(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(
            status_code=400,
            detail="At least one PDF file is required.",
        )

    invalid_files = [
        file.filename
        for file in files
        if not file.filename.lower().endswith(".pdf")
    ]

    if invalid_files:
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are supported. Invalid files: {invalid_files}",
        )

    try:
        result = ingest_pdfs(files)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {e}",
        )

    return MultiIngestResponse(**result)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        return handle_chat(
            request.session_id,
            request.question,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {e}",
        )


@router.get("/health")
async def health():
    return {"status": "ok"}
