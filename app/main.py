from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="RAG Chatbot API",
    description="A Retrieval-Augmented Generation chatbot built with LangChain, "
                 "ChromaDB, and FastAPI.",
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running. See /docs for the API reference."}
