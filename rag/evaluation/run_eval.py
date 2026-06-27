"""
Évaluation RAGAS du pipeline ClinicalRAG, avec LLM juge local (Ollama).

Usage:
    python -m rag.evaluation.run_eval
"""

import os

from datasets import Dataset
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from rag.evaluation.dataset import EVAL_SET
from rag.pipeline import answer_question

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "mistral")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")

RESULTS_PATH = "rag/evaluation/results.csv"

# Config adaptée à l'inférence locale Ollama :
# - timeout long (un LLM 7B local est lent, surtout sans GPU dédié)
# - max_workers=1 : pas de parallélisme, Ollama traite une requête à la fois
#   (le parallélisme par défaut de RAGAS sature la file d'attente et provoque des timeouts)
LOCAL_RUN_CONFIG = RunConfig(
    timeout=600,       # 10 minutes par appel, large marge pour mistral local
    max_workers=1,
    max_retries=2,
    max_wait=60,
)


def build_ragas_dataset() -> Dataset:
    records = []

    for i, item in enumerate(EVAL_SET, start=1):
        question = item["question"]
        print(f"[{i}/{len(EVAL_SET)}] {question}")

        result = answer_question(question)
        contexts = _extract_context_texts(question)

        records.append({
            "question": question,
            "answer": result["answer"],
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        })

    return Dataset.from_list(records)


def _extract_context_texts(question: str) -> list[str]:
    from rag.vectorstore import get_retriever

    retriever = get_retriever(top_k=5, score_threshold=0.5)
    docs = retriever.invoke(question)
    return [doc.page_content for doc in docs]


def main():
    print(f"📊 Évaluation RAGAS — {len(EVAL_SET)} questions\n")
    print("⚠️  LLM juge local (mistral), exécution séquentielle — comptez 20-40 min au total.\n")

    dataset = build_ragas_dataset()

    print("\n🔎 Exécution des métriques RAGAS (séquentiel, timeout 10 min/appel)...\n")

    judge_llm = LangchainLLMWrapper(
        ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=LOCAL_RUN_CONFIG,
    )

    df = result.to_pandas()
    df.to_csv(RESULTS_PATH, index=False)

    print("\n" + "=" * 70)
    print("📊 RÉSULTATS RAGAS")
    print("=" * 70)
    print(df[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]]
          .to_string(index=False))

    print("\n" + "=" * 70)
    print("📈 MOYENNES")
    print("=" * 70)
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"  {metric:20s} {df[metric].mean():.3f}")

    print(f"\n✅ Résultats détaillés sauvegardés → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
