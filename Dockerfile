FROM python:3.11-slim

WORKDIR /app

# Dépendances système minimales (PyMuPDF a besoin de quelques libs natives)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY rag/ rag/
COPY ingestion/ ingestion/

EXPOSE 8000

# OLLAMA_BASE_URL doit pointer vers l'hôte (host.docker.internal sur Docker Desktop)
# puisque Ollama tourne sur la machine hôte, pas dans ce container.
# Le dossier data/vectorstore doit être monté en volume au lancement (trop volumineux
# pour être inclus dans l'image, et déjà exclu du repo via .gitignore).
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
