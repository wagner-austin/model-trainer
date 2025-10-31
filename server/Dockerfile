FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY pyproject.toml /app/
RUN poetry lock --no-interaction --no-ansi --no-update || true \
 && poetry install --no-interaction --no-ansi --no-root

COPY model_trainer /app/model_trainer
RUN poetry install --no-interaction --no-ansi

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "model_trainer.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

