"""
Thin service layer over the RAG chain, converting LangChain Documents
into the API's response schema.
"""
from app.chains.rag_chain import ask
from app.models.schemas import ChatResponse, SourceChunk


def handle_chat(session_id: str, question: str) -> ChatResponse:
    answer, docs = ask(session_id, question)

    sources = [
        SourceChunk(
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page", "?"),
            snippet=doc.page_content[:200].strip() + ("..." if len(doc.page_content) > 200 else ""),
        )
        for doc in docs
    ]

    return ChatResponse(answer=answer, sources=sources, session_id=session_id)
