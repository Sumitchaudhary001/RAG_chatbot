from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client-provided id to track conversation history")
    question: str


class SourceChunk(BaseModel):
    source: str
    page: int | str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    session_id: str


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
    total_chunks_in_store: int

class IngestedDocument(BaseModel):
    filename: str
    chunks_indexed: int


class MultiIngestResponse(BaseModel):
    files_processed: int
    documents: list[IngestedDocument]
    total_new_chunks: int
    total_chunks_in_store: int
