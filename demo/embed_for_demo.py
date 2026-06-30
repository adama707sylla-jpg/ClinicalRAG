"""
Re-génère les embeddings pour la démo publique, avec sentence-transformers
(BAAI/bge-m3 en version HuggingFace) au lieu d'Ollama.

Nécessaire car la démo tournera sur un hébergement sans Ollama (HF Spaces).
Repart de data/processed/chunks.json — pas besoin de re-scraper PubMed/guidelines.

Usage (à exécuter une seule fois, en local, avant de déployer) :
    python demo/embed_for_demo.py
"""

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNKS_PATH = Path("data/processed/chunks.json")
DEMO_VECTORSTORE_PATH = Path("demo/vectorstore")
COLLECTION_NAME = "clinicalrag_demo"
BATCH_SIZE = 32


def main():
    print("📦 Chargement des chunks...")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   → {len(chunks)} chunks à vectoriser\n")

    print("🔌 Chargement du modèle BAAI/bge-m3 (HuggingFace, peut prendre 1-2 min)...")
    model = SentenceTransformer("BAAI/bge-m3")

    DEMO_VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(DEMO_VECTORSTORE_PATH), settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print("\n🔎 Vectorisation par lots (CPU, batch natif — plus rapide que l'API Ollama)...")
    for batch_start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Vectorisation"):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]

        texts = [chunk["text"] for chunk in batch]
        ids = [f"{chunk['source_type']}_{batch_start + i}" for i, chunk in enumerate(batch)]
        metadatas = [
            {
                **{k: v for k, v in chunk["metadata"].items() if isinstance(v, (str, int, float, bool))},
                "source_type": chunk["source_type"],
            }
            for chunk in batch
        ]

        embeddings = model.encode(texts, normalize_embeddings=True).tolist()

        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    print(f"\n✅ TERMINÉ — {collection.count()} chunks vectorisés → {DEMO_VECTORSTORE_PATH}")


if __name__ == "__main__":
    main()
