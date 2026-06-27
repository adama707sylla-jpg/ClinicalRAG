"""
Pipeline RAG complet : retrieval + génération + citations.

Usage:
    from rag.pipeline import answer_question
    result = answer_question("Quel traitement pour l'hypertension ?")
    print(result["answer"])
    print(result["sources"])
"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama

from rag.prompt import RAG_PROMPT
from rag.vectorstore import get_retriever

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")

TOP_K = 5
SCORE_THRESHOLD = 0.5


def build_reference(doc: Document, index: int) -> str:
    meta = doc.metadata
    source_type = meta.get("source_type", "?")

    if source_type == "pubmed":
        return (
            f"[{index}] PubMed — \"{meta.get('title', '')}\" "
            f"({meta.get('journal', '')}, {meta.get('year', '')}) — {meta.get('source_url', '')}"
        )
    else:
        return (
            f"[{index}] OMS — \"{meta.get('title', '')}\" "
            f"(chunk #{meta.get('chunk_index', '?')}) — {meta.get('source_url', '')}"
        )


def format_context(docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        blocks.append(f"[{i}] {doc.page_content}")
    return "\n\n".join(blocks)


def get_llm() -> ChatOllama:
    return ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)


def build_chain():
    retriever = get_retriever(top_k=TOP_K, score_threshold=SCORE_THRESHOLD)
    llm = get_llm()

    retrieval_chain = RunnableParallel(
        docs=retriever,
        question=RunnablePassthrough(),
    )

    def assemble_prompt_inputs(inputs: dict) -> dict:
        num_sources = len(inputs["docs"])
        return {
            "context": format_context(inputs["docs"]),
            "question": inputs["question"],
            "num_sources": num_sources,
            "num_sources_plus_one": num_sources + 1,
        }

    generation_chain = (
        assemble_prompt_inputs
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return retrieval_chain, generation_chain


def answer_question(question: str) -> dict:
    retrieval_chain, generation_chain = build_chain()

    retrieval_result = retrieval_chain.invoke(question)
    docs = retrieval_result["docs"]

    if not docs:
        return {
            "answer": "Aucune source pertinente n'a été trouvée dans la base documentaire "
                       "pour répondre à cette question.",
            "sources": [],
            "num_sources": 0,
        }

    answer = generation_chain.invoke(retrieval_result)
    sources = [build_reference(doc, i) for i, doc in enumerate(docs, start=1)]

    return {
        "answer": answer,
        "sources": sources,
        "num_sources": len(docs),
    }
