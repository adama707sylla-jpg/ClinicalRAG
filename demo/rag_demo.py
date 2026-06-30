"""
Pipeline RAG pour la démo publique (Groq + sentence-transformers + ChromaDB).

Version allégée et autonome du pipeline local (Ollama) :
- Embeddings : BAAI/bge-m3 via sentence-transformers (CPU pur)
- LLM génération : Groq (llama-3.3-70b-versatile)
- Vector store : ChromaDB locale (demo/vectorstore)
"""

import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEMO_VECTORSTORE_PATH = os.getenv("DEMO_VECTORSTORE_PATH", str(Path(__file__).parent / "vectorstore"))
COLLECTION_NAME = "clinicalrag_demo"
TOP_K = 5
SCORE_THRESHOLD = 0.85

_model = None
_collection = None
_groq_client = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-m3")
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=DEMO_VECTORSTORE_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_groq():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def retrieve(query: str) -> list[dict]:
    """Récupère les chunks les plus pertinents pour une question."""
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Filtre par seuil de distance (équivalent du SCORE_THRESHOLD local)
        if dist <= SCORE_THRESHOLD:
            docs.append({"text": doc, "metadata": meta, "distance": dist})

    return docs


def build_reference(meta: dict, index: int) -> str:
    """Formate la référence d'une source pour l'affichage."""
    source_type = meta.get("source_type", "?")
    if source_type == "pubmed":
        return (
            f"[{index}] PubMed — \"{meta.get('title', '')}\" "
            f"({meta.get('journal', '')}, {meta.get('year', '')}) "
            f"— {meta.get('source_url', '')}"
        )
    else:
        return (
            f"[{index}] OMS — \"{meta.get('title', '')}\" "
            f"(chunk #{meta.get('chunk_index', '?')}) "
            f"— {meta.get('source_url', '')}"
        )


def answer(question: str) -> dict:
    """Pipeline RAG complet : retrieve → generate → cite."""
    docs = retrieve(question)

    if not docs:
        return {
            "answer": (
                "Aucune source pertinente n'a été trouvée dans la base documentaire "
                "pour répondre à cette question."
            ),
            "sources": [],
            "num_sources": 0,
        }

    num_sources = len(docs)
    context = "\n\n".join(
        f"[{i}] {doc['text']}" for i, doc in enumerate(docs, start=1)
    )

    system_prompt = f"""Tu es un assistant de recherche clinique. Tu réponds aux questions \
médicales UNIQUEMENT à partir des extraits fournis dans le contexte ci-dessous.

RÈGLES STRICTES :
1. N'utilise QUE les informations du contexte. N'invente JAMAIS.
2. Si le contexte est insuffisant, dis-le clairement.
3. Le contexte contient EXACTEMENT {num_sources} extraits, numérotés [1] à [{num_sources}].
4. OBLIGATOIRE : chaque phrase ET chaque puce de liste doit se terminer par [n].
   Exemple (sujet sans rapport) : "La fermentation nécessite une hydratation contrôlée [1]. \
Les étapes sont :
- le pétrissage [2]
- le pointage [3]"
5. Ne cite JAMAIS un numéro > {num_sources}.
6. NE GÉNÈRE PAS de section Références à la fin.
7. Réponds dans la langue de la question.

CONTEXTE ({num_sources} extraits) :
{context}"""

    groq_client = _get_groq()
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    answer_text = response.choices[0].message.content
    sources = [build_reference(doc["metadata"], i) for i, doc in enumerate(docs, start=1)]

    return {
        "answer": answer_text,
        "sources": sources,
        "num_sources": num_sources,
    }
