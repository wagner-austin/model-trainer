# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Ensure Poetry venv binaries are on PATH
ENV PATH="/app/.venv/bin:${PATH}"

# Copy lockfiles first for better layer caching (root context -> server/*)
COPY server/pyproject.toml /app/pyproject.toml
COPY server/poetry.lock /app/poetry.lock

# Install deps (main only) into in-project venv like handwriting-ai
# Use --no-root here (code not yet copied) to avoid project install error
RUN poetry config virtualenvs.create true \
 && poetry config virtualenvs.in-project true \
 && poetry install --only main --no-interaction --no-ansi --no-root

# Copy the server sources last
COPY server/ /app/

# Install the project itself so console scripts (modeltrainer-rq-worker) are available
RUN poetry install --only main --no-interaction --no-ansi

EXPOSE 8000

# Default to API; Railway worker service should override Start Command
CMD ["/bin/sh", "-lc", \
     "/app/.venv/bin/uvicorn model_trainer.api.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
