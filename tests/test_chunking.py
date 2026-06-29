"""Tests unitaires pour la logique de chunking (ne nécessitent ni Ollama ni ChromaDB)."""

from rag.chunking import CHUNK_OVERLAP, CHUNK_SIZE


def test_chunk_size_is_positive():
    assert CHUNK_SIZE > 0


def test_chunk_overlap_smaller_than_chunk_size():
    """L'overlap doit toujours être strictement inférieur à la taille du chunk,
    sinon le découpage entrerait dans une boucle infinie ou dupliquerait tout."""
    assert CHUNK_OVERLAP < CHUNK_SIZE
