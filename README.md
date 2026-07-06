# 🩺 ClinicalRAG

[![CI](https://github.com/adama707sylla-jpg/ClinicalRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/adama707sylla-jpg/ClinicalRAG/actions/workflows/ci.yml)
[![CD](https://github.com/adama707sylla-jpg/ClinicalRAG/actions/workflows/cd.yml/badge.svg)](https://github.com/adama707sylla-jpg/ClinicalRAG/actions/workflows/cd.yml)
[![Démo en ligne](https://img.shields.io/badge/🤗%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/adama-sylla77/ClinicalRAG)

Assistant de recherche clinique basé sur **RAG** (Retrieval-Augmented Generation), répondant à des questions médicales à partir d'un corpus documentaire vérifiable — articles **PubMed** et guidelines de l'**OMS** — avec citation systématique des sources.

> ⚠️ **Avertissement** : ClinicalRAG est un outil de recherche bibliographique destiné aux professionnels de santé, chercheurs et étudiants en médecine. Ce n'est **pas** un dispositif médical, et il ne fournit ni diagnostic ni recommandation de traitement.

**[→ Essayer la démo en ligne](https://huggingface.co/spaces/adama-sylla77/ClinicalRAG)**

---

## 📊 Le corpus en chiffres

| | |
|---|---|
| Articles PubMed | **2 690** |
| Spécialités médicales couvertes | **14** |
| Guidelines OMS | **6** |
| Chunks vectorisés | **3 475** |

Spécialités couvertes : endocrinologie, cardiologie, pneumologie, psychiatrie, infectiologie, oncologie, neurologie, gynécologie-obstétrique, néphrologie, rhumatologie, dermatologie, médecine d'urgence, gastro-entérologie, hématologie, pédiatrie.

## 🏗️ Architecture

Le projet comprend **deux pipelines distincts**, partageant le même corpus mais adaptés à des contextes d'exécution différents :

| | Pipeline local (développement) | Pipeline démo (production) |
|---|---|---|
| Embeddings | Ollama (`bge-m3`) | `sentence-transformers` (`multilingual-e5-small`) |
| LLM | Ollama (`mistral` 7B) | Groq API (`llama-3.3-70b-versatile`) |
| Vector store | ChromaDB local (`data/vectorstore/`) | ChromaDB (`demo/vectorstore/`, hébergé sur HF Dataset) |
| Interface | API FastAPI | Streamlit (déployé sur Hugging Face Spaces) |
| Cas d'usage | Développement, évaluation RAGAS | Démo publique, sans dépendance à un serveur Ollama |

La bascule vers un modèle d'embedding plus léger pour la démo (`multilingual-e5-small` au lieu de `bge-m3`) a permis un gain de performance d'environ **9,5x** sur le temps de traitement CPU, tout en conservant le support multilingue FR/EN nécessaire aux deux corpus de requêtes.

## ⚙️ Stack technique

- **Ingestion** : NCBI Entrez API (PubMed), scraping structuré des guidelines OMS
- **Chunking** : 1 chunk = 1 article pour PubMed ; découpage à taille fixe avec overlap pour les guidelines (`langchain_text_splitters`)
- **RAG** : LangChain, garde-fous anti-hallucination, citations `[n]` obligatoires
- **Vector store** : ChromaDB (version figée `0.5.5`)
- **API** : FastAPI (`api/main.py`)
- **CI/CD** : GitHub Actions — lint (`ruff`), formatage (`black`), tests (`pytest`), build & push Docker vers GHCR
- **Déploiement démo** : Docker sur Hugging Face Spaces

## 🚀 Démarrage rapide

### Installation

git clone https://github.com/adama707sylla-jpg/ClinicalRAG.git
cd ClinicalRAG
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env

### Reconstruire le corpus complet (pipeline local)

python -m ingestion.pubmed.fetch
python -m rag.chunking
python -m rag.embed

### Lancer l'API

uvicorn api.main:app --reload --port 8000

### Lancer les tests

pytest tests/ -v -m "not requires_ollama"

## 📈 Évaluation

Évaluation RAGAS réalisée sur le pipeline local (corpus initial, 811 articles / 8 thématiques — réévaluation prévue suite à l'extension du corpus) :

| Métrique | Score |
|---|---|
| Faithfulness | 1.0 |
| Context precision | 1.0 |
| Context recall | 0.875 |
| Answer relevancy | 0.916 |

## 📁 Structure du projet

ClinicalRAG/
├── api/                 # API FastAPI
├── data/                # Données brutes/traitées + vector store local
├── demo/                # Pipeline d'embedding pour la démo publique
├── infra/terraform/     # IaC Azure (écrit, non déployé)
├── ingestion/
│   ├── pubmed/          # Ingestion NCBI Entrez
│   └── guidelines/      # Ingestion guidelines OMS
├── notebooks/           # Exploration
├── rag/                 # Chunking, embeddings, vectorstore, évaluation
└── tests/               # Tests unitaires

## 🔗 Liens

- **Démo publique** : [huggingface.co/spaces/adama-sylla77/ClinicalRAG](https://huggingface.co/spaces/adama-sylla77/ClinicalRAG)
- **Auteur** : Adama Sylla — [LinkedIn](https://linkedin.com/in/adama-sylla77) · [Portfolio](https://adasy.netlify.app)

## Licence

Non définie pour l'instant. Merci de me contacter avant toute réutilisation.

