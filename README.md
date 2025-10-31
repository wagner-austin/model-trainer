# Model Trainer

A strictly typed, modular system for training and evaluating small language models and tokenizers on CPU, with an optional path to cloud/scale. It emphasizes reliability, observability, and maintainability: no use of `Any`, no casts, explicit contracts, structured logging, durable jobs via Redis + RQ, and deterministic artifacts/manifests.

## Features
- Pluggable backends via Protocols and registries
  - Tokenizers: BPE (HF Tokenizers) with stats + manifest
  - Models: Tiny GPT‑2 (Transformers + typed PyTorch loop), eval metrics
- Durable job execution: Redis + RQ, heartbeats, cancel flag support
- Structured JSON logging and per-run logs.jsonl
- Clean API surface (FastAPI) for tokenizers, runs, artifacts
- Strict typing, guards, and tests to prevent drift

## Requirements
- Windows + PowerShell (primary), macOS/Linux also supported
- Docker Desktop for containers
- Python 3.11 + Poetry for local lint/test

## Quickstart
1) From `Model-Trainer/` copy environment
   - `cp .env.example .env` (PowerShell: `copy .env.example .env`)
2) Start the stack
   - `make start`
3) Health
   - `GET http://localhost:8000/healthz`
   - `GET http://localhost:8000/readyz`

## Make targets (PowerShell)
- `make lint` – ruff --fix, ruff format, mypy strict, then guard checks
- `make test` – runs pytest for server tests
- `make start` – docker compose up -d --build (Redis, API, Worker)
- `make stop` – docker compose stop
- `make restart` – stop then start
- `make clean` – removes this stack’s containers/images/volumes and restarts

## Configuration (.env)
Key envs (nested via `__` are supported):
- `REDIS_URL=redis://redis:6379/0`
- `RQ__QUEUE_NAME=training`, timeouts/TTL/retry via other `RQ__*`
- `APP__ARTIFACTS_ROOT=/data/artifacts`, `APP__RUNS_ROOT=/data/runs`, `APP__LOGS_ROOT=/data/logs`
- `HF_HOME=/hf-cache` is set in compose for model/tokenizer cache

See `Model-Trainer/.env.example` for defaults.

## API Endpoints
- Tokenizers
  - `POST /tokenizers/train` – enqueue BPE training
  - `GET /tokenizers/{id}` – status + stats + artifact path
- Runs (training + eval)
  - `POST /runs/train` – enqueue training
  - `GET /runs/{run_id}` – status + heartbeat
  - `POST /runs/{run_id}/evaluate` – enqueue eval
  - `GET /runs/{run_id}/eval` – latest eval summary
  - `GET /runs/{run_id}/logs?tail=200` – tail per‑run logs.jsonl
- Artifacts
  - `GET /artifacts/{kind}/{item_id}` – list files
  - `GET /artifacts/{kind}/{item_id}/download?path=...` – download file (supports HTTP Range)
- Health
  - `GET /healthz`, `GET /readyz`

## Job Execution
- Queue: Redis + RQ with retry/timeouts/TTLs
- Worker processes:
  - Training: heartbeats (`runs:hb:<id>`), cancellation via `runs:<id>:cancelled=1`, per-run logs at `artifacts/models/<id>/logs.jsonl`
  - Tokenizer: status and stats persisted in Redis + artifacts folder

## Artifacts & Layout
- `artifacts/tokenizers/<tokenizer_id>/`
  - `tokenizer.json`, `manifest.json`, `logs.jsonl`
- `artifacts/models/<run_id>/`
  - `pytorch_model.bin`, `manifest.json`, `logs.jsonl`, `eval/metrics.json`

## Typing & Quality Gates
- Python: `mypy --strict` with no `Any`, no casts
- Ruff: bans `typing.cast`; stylistic + import rules applied
- Guard checks: blocks `typing.Any`, `typing.cast`, `type: ignore`, and drift markers
- Tests: focused unit/integration tests for tokenizer, dataset, training, eval, artifacts

## Architecture (server)
- `model_trainer/api/*` – FastAPI app + routes
- `model_trainer/core/contracts/*` – Protocols (Model, Tokenizer, Dataset, Compute)
- `model_trainer/core/services/*` – DI container, registries, logging service, dataset builders
- `model_trainer/core/services/tokenizer/*` – BPE backend
- `model_trainer/core/services/training/*` – GPT‑2 backend + dataset builder
- `model_trainer/orchestrators/*` – Orchestrators for enqueue/status
- `model_trainer/worker/*` – RQ workers for training/eval/tokenizer
- `model_trainer/infra/*` – persistence/storage helpers
- `tests/*` – server tests

## Development Flow
- Lint + guards: `make lint`
- Run tests: `make test`
- Start stack: `make start`
- Use API to kick off tokenizer + run; observe logs via `/runs/{id}/logs`

## Notes
- CPU-only MVP, tiny GPT‑2 config; use GPU by swapping compute provider in a future iteration
- Optional Discord DM status bot and UI are deferred; design supports adding them later via contracts

