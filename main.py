from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.core.config import settings
from app.services.heuristic_engine import HeuristicEvaluationEngine
from app.services.omniparser_client import OmniParserClient
from app.services.rag_knowledge_base import RAGKnowledgeBase
from app.api.routes import heuristic, evaluation, health
from app.utils.logging_config import setup_logging

app = FastAPI(
    title="AI Heuristic Evaluation API",
    description="AI-powered heuristic evaluation system using OmniParser and LLMs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081", "http://127.0.0.1:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(heuristic.router, prefix="/api/v1/heuristic", tags=["heuristic"])
app.include_router(evaluation.router, prefix="/api/v1/evaluation", tags=["evaluation"])

@app.on_event("startup")
async def startup_event():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("AI Heuristic Evaluation API starting up...")

    # Initialize OmniParser Client (Singleton)
    app.state.omniparser_client = OmniParserClient()
    await app.state.omniparser_client.initialize()
    logger.info("OmniParser client initialized (singleton)")

    # Initialize RAG Knowledge Base (Singleton)
    app.state.rag_knowledge_base = RAGKnowledgeBase()
    await app.state.rag_knowledge_base.initialize()
    logger.info("RAG knowledge base initialized")

    # Initilize Hueristic Evaluation Engine (Singleton)
    app.state.heuristic_engine = HeuristicEvaluationEngine(
        rag_kb=app.state.rag_knowledge_base
    )
    await app.state.heuristic_engine.initialize()
    logger.info("Heuristic evaluation engine initialized")


@app.on_event("shutdown")
async def shutdown_event():
    logger = logging.getLogger(__name__)
    logger.info("AI Heuristic Evaluation API shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
