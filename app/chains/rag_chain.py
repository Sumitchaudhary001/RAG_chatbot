"""
The core RAG chain, built with LCEL (LangChain Expression Language)
and wrapped with RunnableWithMessageHistory for conversational memory.

Part 2 (RAG pipeline):
    retriever -> format context -> prompt -> LLM -> parse output

Part 3 (memory):
    We use RunnableWithMessageHistory + an in-memory per-session store
    (ChatMessageHistory) rather than the older ConversationBufferMemory
    classes, which are deprecated in modern LangChain. To avoid
    unbounded growth we trim history to the last N messages
    (settings.MAX_HISTORY_MESSAGES) every time it's read, using
    trim_messages. This bounds the number of tokens sent to the LLM
    per turn regardless of how long the conversation runs.
"""
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_groq import ChatGroq

from app.prompts.rag_prompt import rag_prompt, format_context
from app.retrievers.retriever import get_retriever
from app.utils.config import settings

# --- Per-session in-memory chat history store -----------------------------
# In production this would be swapped for Redis/Postgres-backed history
# (RunnableWithMessageHistory supports any BaseChatMessageHistory
# implementation), but in-memory is sufficient for a single-process demo.
_session_store: dict[str, InMemoryChatMessageHistory] = {}


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


def _trim_history(messages):
    """Keep only the most recent N messages to bound context size."""
    return trim_messages(
        messages,
        max_tokens=settings.MAX_HISTORY_MESSAGES,
        token_counter=len,          # counts messages, not tokens
        strategy="last",
        start_on="human",
        include_system=False,
    )


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model=settings.CHAT_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=0,
    )


def _retrieve_and_remember(inputs: dict) -> dict:
    """Runs retrieval and stashes the retrieved docs for source citation."""
    retriever = get_retriever()
    docs: List[Document] = retriever.invoke(inputs["question"])
    inputs["_retrieved_docs"] = docs
    inputs["context"] = format_context(docs)
    return inputs


def build_rag_chain():
    """
    Builds the full LCEL pipeline:

    {question, history} -> retrieve -> format context -> prompt -> LLM -> text
    """
    llm = _get_llm()

    chain = (
        RunnablePassthrough.assign(history=lambda x: _trim_history(x["history"]))
        | RunnableLambda(_retrieve_and_remember)
        | RunnablePassthrough.assign(
            answer=rag_prompt | llm | StrOutputParser()
        )
    )

    chain_with_memory = RunnableWithMessageHistory(
        chain,
        _get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    return chain_with_memory


_rag_chain = None


def get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = build_rag_chain()
    return _rag_chain


def ask(session_id: str, question: str) -> Tuple[str, List[Document]]:
    """Run one turn of the RAG chatbot and return (answer, source_docs)."""
    chain = get_rag_chain()
    result = chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )
    return result["answer"], result.get("_retrieved_docs", [])
