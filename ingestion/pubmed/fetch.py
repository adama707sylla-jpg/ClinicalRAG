"""
Script de récupération des articles PubMed via l'API NCBI Entrez.

Pour chaque thématique définie dans topics.py :
1. esearch : récupère les PMIDs correspondant à la requête
2. efetch  : récupère les métadonnées complètes (titre, abstract, auteurs, etc.)
3. Sauvegarde en JSON dans ingestion/pubmed/raw/{topic_name}.json

Usage:
    python -m ingestion.pubmed.fetch
"""

import json
import os
import time
from pathlib import Path

from Bio import Entrez
from dotenv import load_dotenv
from tqdm import tqdm

from ingestion.pubmed.topics import TOPICS

load_dotenv()

Entrez.email = os.getenv("NCBI_EMAIL")
if not Entrez.email:
    raise ValueError("NCBI_EMAIL manquant dans .env")

NCBI_API_KEY = os.getenv("NCBI_API_KEY")
if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Sans clé API : max 3 requêtes/seconde. Avec clé : 10/seconde.
SLEEP_BETWEEN_CALLS = 0.4 if not NCBI_API_KEY else 0.12


def search_pmids(query: str, max_results: int) -> list[str]:
    """Récupère la liste des PMIDs correspondant à une requête Entrez."""
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def fetch_articles(pmids: list[str]) -> list[dict]:
    """Récupère les métadonnées complètes pour une liste de PMIDs, par lots de 50."""
    articles = []
    batch_size = 50
    batch_starts = list(range(0, len(pmids), batch_size))

    for i in tqdm(batch_starts, desc="   Téléchargement (lots de 50)"):
        batch = pmids[i : i + batch_size]
        handle = Entrez.efetch(db="pubmed", id=batch, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        for record in records.get("PubmedArticle", []):
            article = parse_article(record)
            if article:
                articles.append(article)

        time.sleep(SLEEP_BETWEEN_CALLS)

    return articles


def parse_article(record: dict) -> dict | None:
    """Extrait les champs utiles d'un enregistrement PubMed brut."""
    try:
        medline = record["MedlineCitation"]
        article = medline["Article"]

        pmid = str(medline["PMID"])
        title = str(article.get("ArticleTitle", ""))

        abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
        abstract = " ".join(str(part) for part in abstract_parts)

        authors = []
        for author in article.get("AuthorList", []):
            last = author.get("LastName", "")
            fore = author.get("ForeName", "")
            if last:
                authors.append(f"{fore} {last}".strip())

        journal = str(article.get("Journal", {}).get("Title", ""))

        pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year", pub_date.get("MedlineDate", "Unknown"))

        mesh_terms = []
        for mesh in medline.get("MeshHeadingList", []):
            descriptor = mesh.get("DescriptorName", "")
            if descriptor:
                mesh_terms.append(str(descriptor))

        if not abstract:
            return None  # On ignore les articles sans abstract (peu utiles pour le RAG)

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "journal": journal,
            "year": year,
            "mesh_terms": mesh_terms,
            "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }
    except (KeyError, IndexError) as e:
        print(f"⚠️  Erreur parsing article: {e}")
        return None


def main():
    print(f"🔬 Récupération PubMed — {len(TOPICS)} thématiques\n")

    summary = []

    for topic in TOPICS:
        name = topic["name"]
        query = topic["query"]
        max_results = topic["max_results"]

        print(f"📂 {name} — requête: {query}")

        pmids = search_pmids(query, max_results)
        print(f"   → {len(pmids)} PMIDs trouvés")

        if not pmids:
            print(f"   ⚠️  Aucun résultat, on passe au suivant\n")
            continue

        articles = fetch_articles(pmids)

        output_path = RAW_DIR / f"{name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

        print(f"   ✅ {len(articles)} articles sauvegardés → {output_path}\n")
        summary.append({"topic": name, "count": len(articles)})

        time.sleep(SLEEP_BETWEEN_CALLS)

    total = sum(s["count"] for s in summary)
    print(f"\n{'='*50}")
    print(f"✅ TERMINÉ — {total} articles récupérés au total")
    print(f"{'='*50}")
    for s in summary:
        print(f"  {s['topic']:30s} {s['count']:4d} articles")


if __name__ == "__main__":
    main()
