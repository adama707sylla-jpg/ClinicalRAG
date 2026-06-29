"""
Génération des embeddings et stockage dans ChromaDB.

Pour chaque chunk (PubMed + guidelines) :
1. Génère l'embedding via Ollama (nomic-embed-text)
2. Stocke le vecteur + texte + métadonnées dans une collection ChromaDB persistante

Usage:
    python -m rag.embed
"""

import json
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm

load_dotenv()

CHUNKS_PATH = Path("data/processed/chunks.json")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vectorstore")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

COLLECTION_NAME = "clinicalrag_chunks"
BATCH_SIZE = 32  # nombre de chunks traités avant chaque insertion ChromaDB


def load_chunks() -> list[dict]:
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_chroma_collection():
    """Initialise (ou récupère) la collection ChromaDB persistante."""
    client = chromadb.PersistentClient(
        path=VECTOR_STORE_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(name=COLLECTION_NAME)


def flatten_metadata(metadata: dict) -> dict:
    """
    ChromaDB n'accepte que des types simples (str, int, float, bool) dans les métadonnées.
    On s'assure qu'aucune valeur complexe (liste, dict imbriqué) ne s'y glisse.
    """
    flat = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            flat[key] = str(value)
    return flat


def main():
    chunks = load_chunks()
    print(f"📦 {len(chunks)} chunks à vectoriser\n")

    print(
        f"🔌 Connexion à Ollama ({OLLAMA_BASE_URL}, modèle: {OLLAMA_EMBEDDING_MODEL})"
    )
    embedder = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    collection = get_chroma_collection()
    existing_count = collection.count()
    if existing_count > 0:
        print(
            f"⚠️  La collection contient déjà {existing_count} chunks. "
            f"Ils seront ignorés s'ils ont le même ID (pas de doublons).\n"
        )

    for batch_start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Vectorisation"):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]

        texts = [chunk["text"] for chunk in batch]
        ids = [
            f"{chunk['source_type']}_{batch_start + i}" for i, chunk in enumerate(batch)
        ]
        metadatas = [
            {**flatten_metadata(chunk["metadata"]), "source_type": chunk["source_type"]}
            for chunk in batch
        ]

        embeddings = embedder.embed_documents(texts)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    final_count = collection.count()
    print(f"\n{'='*50}")
    print(f"✅ TERMINÉ — {final_count} chunks vectorisés dans ChromaDB")
    print(f"📁 Stockage persistant: {VECTOR_STORE_PATH}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
