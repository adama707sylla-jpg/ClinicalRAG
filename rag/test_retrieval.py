"""
Test manuel du retrieval : interroge ChromaDB avec une question médicale
et affiche les chunks les plus pertinents, pour valider la qualité
de la recherche vectorielle avant de construire le pipeline RAG complet.

Usage:
    python -m rag.test_retrieval "ta question ici"
"""

import os
import sys

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

load_dotenv()

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vectorstore")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
COLLECTION_NAME = "clinicalrag_chunks"


def retrieve(query: str, top_k: int = 5) -> dict:
    embedder = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    query_embedding = embedder.embed_query(query)

    client = chromadb.PersistentClient(
        path=VECTOR_STORE_PATH, settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results


def display_results(query: str, results: dict):
    print(f"\n🔍 Question: {query}\n")
    print("=" * 70)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
        source = meta.get("source_type", "?")
        topic = meta.get("topic", "?")

        if source == "pubmed":
            ref = f"PMID {meta.get('pmid')} — {meta.get('title', '')[:60]}"
        else:
            ref = f"{meta.get('title', '')[:60]} (chunk #{meta.get('chunk_index')})"

        print(f"\n#{rank} [{source.upper()} / {topic}] — distance: {distance:.4f}")
        print(f"Source: {ref}")
        print(f"Extrait: {doc[:250]}...")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m rag.test_retrieval \"ta question\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    results = retrieve(query, top_k=5)
    display_results(query, results)
