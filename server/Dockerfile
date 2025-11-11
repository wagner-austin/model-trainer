# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV POETRY_VERSION=1.8.3 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

COPY . .

# API image
FROM base AS api
EXPOSE 8000
# Respect platform-provided PORT (Railway, Render, Heroku, etc.) with a sane default
CMD ["/bin/sh", "-lc", "poetry run uvicorn model_trainer.api.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]

# Worker image
FROM base AS worker
CMD ["poetry", "run", "rq", "worker", "training", "--with-scheduler"]
