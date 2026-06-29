"""
Schémas Pydantic pour l'API ClinicalRAG.

Définissent le contrat d'entrée/sortie des endpoints — validation automatique
par FastAPI, et documentation Swagger générée à partir de ces modèles.
"""

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Requête entrante pour poser une question au pipeline RAG."""

    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="La question médicale à poser au système",
        examples=["Quel est le traitement de première intention pour l'hypertension ?"],
    )


class AnswerResponse(BaseModel):
    """Réponse du pipeline RAG, avec sources sourcées."""

    question: str
    answer: str
    sources: list[str]
    num_sources: int


class HealthResponse(BaseModel):
    """Réponse du endpoint de santé."""

    status: str
    vectorstore_chunks: int
    llm_model: str
    embedding_model: str
