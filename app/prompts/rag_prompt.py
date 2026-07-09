"""
Prompt engineering for the RAG chatbot (Part 4).

Design goals:
- Only answer from retrieved context (grounding).
- Say "I don't know" rather than fabricate when context is insufficient.
- Be concise and factual, not conversational filler.
- Reference sources by filename/page so answers are auditable.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """You are a factual question-answering assistant. You must \
answer ONLY using the information contained in the CONTEXT below. Do not use \
any outside knowledge, and do not guess.

Rules:
1. If the CONTEXT does not contain enough information to answer the \
question, respond exactly with: "I don't know based on the provided documents."
2. Keep answers concise and factual -- no filler, no speculation.
3. When you use a fact from the context, cite it inline like [source: \
filename, page X].
4. Do not fabricate sources, page numbers, or facts that are not present \
in the CONTEXT.
5. You may use the conversation history to understand follow-up questions \
(e.g. "what about part 2?"), but the ANSWER itself must still come only \
from CONTEXT.

CONTEXT:
{context}
"""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


def format_context(docs) -> str:
    """Turn retrieved Documents into a citation-friendly context block."""
    blocks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[source: {source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks) if blocks else "No relevant context found."
