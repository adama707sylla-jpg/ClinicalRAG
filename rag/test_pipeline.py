"""
Test interactif du pipeline RAG complet.

Usage:
    python -m rag.test_pipeline "ta question ici"
"""

import sys

from rag.pipeline import answer_question


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m rag.test_pipeline "ta question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    print(f"\n🔍 Question: {question}\n")
    print("⏳ Génération en cours...\n")

    result = answer_question(question)

    print("=" * 70)
    print("💬 RÉPONSE:")
    print("=" * 70)
    print(result["answer"])
    print()
    print("=" * 70)
    print(f"📚 SOURCES ({result['num_sources']}):")
    print("=" * 70)
    for source in result["sources"]:
        print(source)


if __name__ == "__main__":
    main()
