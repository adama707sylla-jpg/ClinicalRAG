"""
Stratégie de chunking adaptée par type de source pour ClinicalRAG.

- PubMed (abstracts) : 1 chunk = 1 article entier (déjà court et auto-contenu)
- Guidelines OMS (PDF longs) : découpage par taille fixe avec overlap,
  basé sur le texte complet (full_text) de chaque document

Chaque chunk produit un dict avec :
    - text       : le contenu du chunk
    - source_type: "pubmed" ou "guideline"
    - metadata   : infos de traçabilité (pour les citations dans les réponses du RAG)
"""

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

PUBMED_RAW_DIR = Path("ingestion/pubmed/raw")
GUIDELINES_PROCESSED_DIR = Path("ingestion/guidelines/processed")

# Paramètres du découpage pour les guidelines (en caractères, pas en tokens,
# pour rester simple — ~4 caractères ≈ 1 token en anglais/français)
CHUNK_SIZE = 1500       # ≈ 375 tokens, bon compromis contexte/précision
CHUNK_OVERLAP = 200     # ≈ 50 tokens, évite de couper une idée en plein milieu


def chunk_pubmed_articles() -> list[dict]:
    """1 chunk = 1 article PubMed (titre + abstract concaténés)."""
    chunks = []

    for json_file in PUBMED_RAW_DIR.glob("*.json"):
        topic_name = json_file.stem
        with open(json_file, encoding="utf-8") as f:
            articles = json.load(f)

        for article in articles:
            text = f"{article['title']}\n\n{article['abstract']}"
            chunks.append({
                "text": text,
                "source_type": "pubmed",
                "metadata": {
                    "pmid": article["pmid"],
                    "title": article["title"],
                    "journal": article["journal"],
                    "year": article["year"],
                    "topic": topic_name,
                    "source_url": article["source_url"],
                },
            })

    return chunks


def chunk_guidelines() -> list[dict]:
    """Découpage par taille fixe avec overlap pour les guidelines (documents longs)."""
    chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # essaie de couper aux frontières naturelles
    )

    for json_file in GUIDELINES_PROCESSED_DIR.glob("*.json"):
        with open(json_file, encoding="utf-8") as f:
            doc = json.load(f)

        text_chunks = splitter.split_text(doc["full_text"])

        for i, chunk_text in enumerate(text_chunks):
            chunks.append({
                "text": chunk_text,
                "source_type": "guideline",
                "metadata": {
                    "document_name": doc["name"],
                    "title": doc["title"],
                    "topic": doc["topic"],
                    "chunk_index": i,
                    "source_url": doc["source_url"],
                },
            })

    return chunks


def build_all_chunks() -> list[dict]:
    """Construit le corpus complet de chunks (PubMed + Guidelines)."""
    pubmed_chunks = chunk_pubmed_articles()
    guideline_chunks = chunk_guidelines()

    print(f"📚 PubMed   : {len(pubmed_chunks)} chunks (1 par article)")
    print(f"📋 Guidelines: {len(guideline_chunks)} chunks (découpage {CHUNK_SIZE} car., overlap {CHUNK_OVERLAP})")
    print(f"📦 Total    : {len(pubmed_chunks) + len(guideline_chunks)} chunks")

    return pubmed_chunks + guideline_chunks


if __name__ == "__main__":
    chunks = build_all_chunks()

    # Sauvegarde pour inspection / réutilisation par le script d'embeddings
    output_path = Path("data/processed/chunks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Chunks sauvegardés → {output_path}")
