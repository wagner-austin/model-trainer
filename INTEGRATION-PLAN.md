# Model-Trainer ↔ DiscordBot Integration Plan

Purpose
- Integrate Model-Trainer’s training/eval flows with DiscordBot using a clean, typed contract: enqueue via HTTP, publish progress via Redis, and send rich embed DM updates to users.
- Align reliability patterns: strict typing, request correlation, API key auth, deterministic logs/artifacts.

Scope
- Emit training events (started/progress/completed/failed) on Redis channel `trainer:events`.
- Publish events inline from the worker (authoritative source).
- API accepts `user_id` so DiscordBot can DM the initiating user.
- Enforce request correlation (`X-Request-ID`) and API key auth consistently.
- Provide deployment notes for Railway and wiring details for DiscordBot.

References (code)
- DiscordBot
  - Model-Trainer HTTP client: `DiscordBot/src/clubbot/services/modeltrainer/client.py:1`
  - Trainer events and subscriber: `DiscordBot/src/clubbot/services/jobs/trainer_events.py:1`, `DiscordBot/src/clubbot/services/jobs/trainer_notifier.py:1`
  - Slash command (enqueue): `DiscordBot/src/clubbot/cogs/trainer.py:1`
- Model-Trainer
  - App and routers: `Model-Trainer/server/model_trainer/api/main.py:1`, `Model-Trainer/server/model_trainer/api/routes/runs.py:1`
  - Orchestrator + RQ adapter: `Model-Trainer/server/model_trainer/orchestrators/training_orchestrator.py:1`, `Model-Trainer/server/model_trainer/core/services/queue/rq_adapter.py:1`
  - Worker (training/eval): `Model-Trainer/server/model_trainer/worker/training_worker.py:1`
  - Event schema: `Model-Trainer/server/model_trainer/events/trainer.py:1`
  - Error handler: `Model-Trainer/server/model_trainer/core/errors/handlers.py:1`
  - Settings: `Model-Trainer/server/model_trainer/core/config/settings.py:1`

Event Model (implemented)
- Channel: `trainer:events`
- Types: `trainer.train.started.v1`, `trainer.train.progress.v1`, `trainer.train.completed.v1`, `trainer.train.failed.v1`
- Source of truth: worker publishes via `redis.publish` in `training_worker.py`
- Payloads: typed in `events/trainer.py`; `request_id` equals `run_id`

API Contract (implemented)
- POST `/runs/train` with body `TrainRequest` including `user_id: int`
- GET `/runs/{run_id}` returns `RunStatusResponse` with `status`, `last_heartbeat_ts`, and optional `message`
- Logs and cancel endpoints: `/runs/{run_id}/logs`, `/runs/{run_id}/logs/stream`, `/runs/{run_id}/cancel`
- Correlation: `X-Request-ID` echoed on all responses; errors include `code`, `message`, `request_id`
- Auth: API key dependency active when configured; invalid/missing key → HTTP 401 (code `UNAUTHORIZED`)

DiscordBot Wiring
- Uses HTTP client with optional `X-Api-Key` and correlation header.
- Subscribes to `trainer:events` and DMs the target user id carried in events.
- Slash `/train_model` command enqueues training and confirms with run/job IDs.

Railway
- Two services: `api` (uvicorn) and `worker` (rq worker), plugin: `redis` — see `Model-Trainer/DEPLOYING_RAILWAY.md`.
- Use Docker builder with `server/Dockerfile` build targets: `api` and `worker`.
- Env: `REDIS_URL`, `SECURITY__API_KEY`, `APP__ARTIFACTS_ROOT`, `APP__RUNS_ROOT`, `APP__LOGS_ROOT`.

Quality Bar
- mypy strict (no Any, no casts, no ignores) and ruff clean.
- 100% statement and branch coverage; tests for middleware, event encoding, run status, and error mapping.

Open Items (tracked)
- Optional metrics in progress events (throughput/memory).
- Calibrator integration to compute optimal threads/workers and reflect in `started` events.
