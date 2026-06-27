"""
Wrapper LangChain autour de la collection ChromaDB existante.

Permet d'utiliser la base vectorielle déjà peuplée (via rag/embed.py)
comme un Retriever LangChain standard, pour l'intégrer dans une chaîne LCEL.
"""

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vectorstore")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
COLLECTION_NAME = "clinicalrag_chunks"


def get_vectorstore() -> Chroma:
    """Charge la collection ChromaDB persistante comme un VectorStore LangChain."""
    embedder = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedder,
        persist_directory=VECTOR_STORE_PATH,
    )


def get_retriever(top_k: int = 5, score_threshold: float | None = None):
    """
    Construit un retriever LangChain à partir du vector store.

    score_threshold : si fourni, filtre les chunks trop éloignés sémantiquement
    (évite d'envoyer du bruit au LLM — ex: page de licence récupérée par erreur).
    """
    vectorstore = get_vectorstore()

    search_kwargs = {"k": top_k}
    if score_threshold is not None:
        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": top_k, "score_threshold": score_threshold},
        )

    return vectorstore.as_retriever(search_kwargs=search_kwargs)
