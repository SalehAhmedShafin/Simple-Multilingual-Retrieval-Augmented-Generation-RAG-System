from fastapi import FastAPI
from app.routes import rag_routes

app = FastAPI(
    title="Multilingual RAG System",
    description="A simple RAG system for querying a Bengali PDF document in English and Bengali."
)

app.include_router(rag_routes.router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the RAG API.",
        "api_docs": "/docs"
    }