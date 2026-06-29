"""
API FastAPI pour ClinicalRAG.

Expose le pipeline RAG (retrieval + génération + citations) via HTTP.

Usage:
    uvicorn api.main:app --reload --port 8000
"""

import logging
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import AnswerResponse, HealthResponse, QuestionRequest
from rag.pipeline import OLLAMA_LLM_MODEL, answer_question
from rag.vectorstore import OLLAMA_EMBEDDING_MODEL, get_vectorstore

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("clinicalrag")

app = FastAPI(
    title="ClinicalRAG API",
    description=(
        "Assistant de recherche clinique RAG, basé sur la littérature PubMed "
        "et les recommandations officielles OMS, avec citations sourcées."
    ),
    version="0.1.0",
)

# CORS : autorise le frontend (origine différente, ex: localhost:5173 en dev Vite)
# à appeler cette API depuis le navigateur.
FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["health"])
def root():
    return {"message": "ClinicalRAG API — voir /docs pour la documentation interactive"}


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    """Vérifie que la base vectorielle est accessible et renvoie les infos de config."""
    try:
        vectorstore = get_vectorstore()
        chunk_count = vectorstore._collection.count()
    except Exception as e:
        logger.error(f"Vectorstore inaccessible: {e}")
        raise HTTPException(status_code=503, detail="Base vectorielle inaccessible")

    return HealthResponse(
        status="ok",
        vectorstore_chunks=chunk_count,
        llm_model=OLLAMA_LLM_MODEL,
        embedding_model=OLLAMA_EMBEDDING_MODEL,
    )


@app.post("/ask", response_model=AnswerResponse, tags=["rag"])
def ask(request: QuestionRequest):
    """
    Pose une question au pipeline RAG ClinicalRAG.

    Retourne une réponse générée à partir des sources pertinentes (PubMed + OMS),
    avec citations numérotées et liste des sources utilisées.
    """
    start_time = time.time()
    logger.info(f"Question reçue: {request.question}")

    try:
        result = answer_question(request.question)
    except Exception as e:
        logger.exception(f"Erreur pendant le traitement de la question: {e}")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne lors du traitement de la question. "
            "Vérifie que Ollama est démarré et accessible.",
        )

    elapsed = time.time() - start_time
    logger.info(f"Réponse générée en {elapsed:.1f}s — {result['num_sources']} sources")

    return AnswerResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        num_sources=result["num_sources"],
    )
