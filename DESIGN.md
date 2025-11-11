# Model Trainer — Design Document

## 1. Vision & Goals

A modular, robust, and strictly typed system to train and evaluate language models (MVP: GPT‑2) and tokenizers (BPE, with optional SentencePiece) on CPU-only laptops. The system is API‑first today; a simple web UI is planned but not yet present in the repository. It emphasizes reliability, observability, and maintainability. It supports multi-threaded CPU execution for dataset preparation and tokenization, with a future path to cloud compute while remaining fully functional locally.

### Primary Goals
- Pluggable model backends via a standard contract (GPT‑2 implemented; LLaMA, Qwen planned).
- Pluggable tokenizer backends via a standard contract (BPE implemented; SentencePiece optional when binaries are available).
- CPU-first operation with multi-threaded preprocessing/tokenization.
- Strict typing end-to-end; no `Any`, no casts.
- Centralized, categorized, structured logging; zero silent exceptions.
- Explicit dependency injection via a service container.
- Simple web UI for end-to-end training control and monitoring (planned).
- Deterministic, reproducible runs with clear artifacts and manifests.

### Non-Goals (Initial MVP)
- Distributed or GPU training.
- Fine-grained model parallelism or quantization.
- Advanced hyperparameter tuning or AutoML.

### Constraints & Assumptions
- Runs on a laptop CPU; training time may be long; target tiny/small models first.
- Model weight access for certain families (e.g., LLaMA) may require gated download.
- Avoid vendor lock-in. Cloud is optional and deferred.

## 1.1 Current Implementation Snapshot (MVP)

- API-first system (FastAPI) exposing tokenizers, runs, artifacts, and health endpoints.
- Tokenizers: BPE implemented; SentencePiece optional if `spm_*` binaries are available.
- Models: GPT‑2 backend implemented (tiny/small CPU configurations) with train and eval.
- Orchestrators: enqueue training/eval and attach per‑run log files.
- Workers: RQ worker performs training/eval, emits heartbeats, handles cancellation, and publishes typed training events to Redis (`trainer:events`).
- Storage: all run manifests and logs live under `artifacts/models/<run_id>/`.

## 1.2 Project Structure (Current)

```
server/
  model_trainer/
    api/
      main.py
      routes/
        artifacts.py, health.py, runs.py, tokenizers.py
      schemas/
    core/
      config/settings.py
      contracts/
      errors/handlers.py
      infra/paths.py, redis_utils.py
      logging/service.py, setup.py, types.py
      services/
        container.py
        dataset/
        model/backends/
        queue/rq_adapter.py
        tokenizer/bpe_backend.py, spm_backend.py (optional)
    infra/
      storage/run_store.py
      persistence/
    orchestrators/
      training_orchestrator.py, tokenizer_orchestrator.py
    worker/
      training_worker.py
    tests/
```


## 2. User Flow (MVP)
1. Select a model backend (e.g., GPT‑2 Small) and version (via API).
2. Select a tokenizer backend (e.g., BPE) and configure vocab size.
3. Point to a cleaned text corpus path; configure holdout split for tokenizer and validation.
4. Train tokenizer; validate on holdout (coverage, OOV rate, basic stats).
5. Train model on CPU with basic metrics (loss, perplexity).
6. Monitor logs and progress via API endpoints; view run summary and artifacts.
7. Export artifacts (tokenizer files, checkpoints, training manifest) from `artifacts/`.


## 3. High-Level Architecture

- UI (Web, TypeScript): planned; interacts with API via JSON.
- API Server (Python, FastAPI):
  - Exposes training runs, tokenizer building, model selection, and artifact browsing.
  - Hosts orchestrators that call core services via DI container.
- Core Services (Python):
  - Tokenizer service (pluggable backends via contracts).
  - Model training service (pluggable backends via contracts).
  - Data ingestion and dataset service.
  - Compute service (local CPU now; cloud later).
  - Logging service (centralized JSON logging with categories).
  - Error service (typed errors, standardized error codes, global handlers).
- Adapters / Backends:
  - Model backends: GPT‑2 (Transformers) implemented; LLaMA/Qwen planned (unavailable placeholders present).
  - Tokenizer backends: Hugging Face Tokenizers (BPE) implemented; SentencePiece optional (requires `spm_*` binaries).
- Storage & Artifacts:
  - Local directory structure for artifacts (tokenizers, models) and per‑run logs under `artifacts/`.
  - Per‑run manifest for reproducibility at `artifacts/models/<run_id>/manifest.json`.


## 4. Technology Choices

- Language (Core/API): Python 3.11+
  - Strict typing: mypy (strict mode), no `typing.Any`, no `typing.cast`.
  - Pydantic v2 for all data models (configs, DTOs, results). Prefer `BaseModel` with `extra="forbid"` and `validate_assignment=True`. Avoid `@dataclass` for these shapes.
  - FastAPI + Uvicorn for API with typed routes and models.
  - Hugging Face: Transformers, Datasets, Tokenizers.
  - PyTorch (CPU) for training; CPU threading configured explicitly.
  - Ruff + mypy in pre-commit; black/ruff format if adopted by repo.
- Language (UI): planned; not yet implemented.
- Logging: Python `logging` or `structlog` with JSON handlers and standardized fields.
- Testing: pytest + mypy; Playwright or Cypress optional for UI later.
- Queue: Redis + RQ (via `redis-py`) for durable training jobs and status updates.


## 5. Project Structure (Proposed)

```
Model-Trainer/
  DESIGN.md
  README.md (later)
  server/
    pyproject.toml
    model_trainer/
      api/
        routes/               # FastAPI routers (typed)
        schemas/              # Pydantic models (strict)
      core/
        contracts/            # Protocols and typed interfaces
        logging/
        errors/
        config/
        services/
          container.py        # Service container (DI)
          tokenizer/
            backends/         # BPE, SentencePiece adapters
          model/
            backends/         # GPT-2, LLaMA, Qwen adapters
          data/
          training/
          compute/
      infra/
        storage/
        persistence/
      orchestrators/
        tokenizer_orchestrator.py
        training_orchestrator.py
  ui/
    package.json
    src/
      app/                    # Routes
      components/
      lib/                    # Typed API client
```


## 6. Contracts (Strict, Stable Interfaces)

Use Python `typing.Protocol` to define implementation-agnostic contracts. No `Any`; use precise generics and TypedDicts/NamedTuples where useful. Example signatures focus on clarity and future extensibility.

### 6.1 Tokenizer Contracts

- `TokenizerConfig`: Pydantic model including `method`, `vocab_size`, `min_frequency`, normalization options, and `special_tokens`.
- `TokenizerTrainRequest`: input corpus path(s), holdout fraction, seed, threads.
- `TokenizerStats`: coverage, OOV rate, token count, char coverage.
- `TokenizerArtifact`: paths to vocab/merges or model files, manifest.

Protocol:
- `TokenizerBackend`:
  - `name() -> str`
  - `supported_methods() -> set[TokenizerMethod]`
  - `train(config: TokenizerConfig, request: TokenizerTrainRequest) -> TokenizerArtifact`
  - `load(artifact_path: str) -> TokenizerHandle`
  - `inspect(handle: TokenizerHandle) -> TokenizerStats`
  - `encode(handle: TokenizerHandle, text: str) -> list[int]`
  - `decode(handle: TokenizerHandle, ids: list[int]) -> str`

Notes:
- Provide BPE via Hugging Face Tokenizers; SentencePiece via `sentencepiece`.
- Multi-threaded training via library options and parallel corpus streaming.

### 6.6 Data Models Policy (Pydantic-first)
- Use `pydantic.BaseModel` for:
  - API schemas, configuration objects, contract types (e.g., `TokenizerTrainConfig`, `ModelTrainConfig`), and result/DTO types (e.g., `TrainOutcome`, `TokenizerTrainStats`).
- Use `TypedDict` for JSON-like manifests persisted to disk (e.g., training manifest structures).
- Use plain classes (or non-frozen dataclasses) for services/containers where runtime validation is not needed and unit tests may monkeypatch attributes.

Guard enforcement:
- Disallow `@dataclass(frozen=True)` and any `@dataclass` in `core/contracts` and `core/config`.
- Disallow `typing.Any`, `typing.cast`, `type: ignore`.
- Disallow `print()` in library code; allowed in tests.

### 6.2 Model Contracts

- `ModelConfig`: model family, size, max_seq_len, optimizer config, LR schedule, batch sizes, seeds, num_epochs, gradient clipping.
- `TrainingDataConfig`: dataset path(s), tokenizer artifact ref, splits, shuffle buffer, num_workers.
- `TrainerConfig`: checkpoint intervals, eval intervals, early stopping, logging cadence, CPU thread limits.
- `TrainingRunManifest`: exact config, versions, seeds, git commit (if any), start/end timestamps, metrics summary.

Protocol:
- `ModelBackend`:
  - `name() -> str`
  - `supported_tokenizers() -> set[TokenizerMethod]`
  - `prepare(config: ModelConfig, tokenizer: TokenizerHandle) -> PreparedModel`
  - `train(prepared: PreparedModel, data_cfg: TrainingDataConfig, trainer_cfg: TrainerConfig) -> TrainingResult`
  - `evaluate(prepared: PreparedModel, split: str) -> EvalResult`
  - `save(prepared: PreparedModel, out_dir: str) -> ModelArtifact`
  - `load(artifact_path: str, tokenizer: TokenizerHandle) -> PreparedModel`

Notes:
- Initial implementation leverages Hugging Face AutoModelForCausalLM, DataCollator, and Trainer on CPU with careful threading.

### 6.3 Data Contracts
- `CorpusProvider` Protocol: streams normalized text; exposes sizes for train/holdout.
- `DatasetBuilder` Protocol: builds tokenized datasets and returns typed dataset handles compatible with backends.

### 6.4 Compute Contracts
- `ComputeProvider` Protocol:
  - `kind() -> Literal['local-cpu', 'cloud']`
  - `threads() -> int`
  - `env() -> dict[str,str]`
  - For future cloud: job submit/cancel/status.


## 7. Dependency Injection (Service Container)

- A small, explicit DI container with:
  - Registration by typed token (Python `Type` or `Literal` keys).
  - Constructor injection via factories; no global singletons.
  - Explicit lifetime scopes (app, request/run, ephemeral).
- Container composes:
  - LoggingService, ErrorService, ConfigService.
  - TokenizerRegistry (maps names -> `TokenizerBackend`).
  - ModelRegistry (maps names -> `ModelBackend`).
  - DataService, TrainingService, ComputeService.

Benefits:
- Clear boundaries and mocking in tests; easy backend swaps.


## 8. Logging (Centralized, Structured, Categorized)

- JSON lines logger; fields:
  - `timestamp`, `level`, `category`, `service`, `event`, `message`, `run_id`, `context`, `error_code` (if applicable).
- Categories: `core`, `api`, `data`, `tokenizer`, `model`, `training`, `compute`, `ui`.
- Output sinks:
  - Console (dev-friendly), per-run file at `artifacts/models/<id>/logs.jsonl`.
- Logging policy:
  - No prints in code; only logger.
  - Include progress markers (e.g., epochs, steps, evals).


## 9. Error Handling (Strict, No Silent Failures)

- Typed `AppError` with `ErrorCode` enum (e.g., `DATA_NOT_FOUND`, `TOKENIZER_TRAIN_FAILED`, `MODEL_TRAIN_FAILED`, `CONFIG_INVALID`).
- API layer maps `AppError` to HTTP error responses with structured JSON body.
- Orchestrators catch known exceptions, convert to `AppError`, log with context, re-raise.
- Unknown exceptions bubble to a global handler that logs with stack and returns a sanitized error body.

Policy (enforced by guard):
- Broad catches (`except:`, `except Exception`, `except BaseException`) must both log (error/exception) and re-raise.
- Specific exception catches must at least log or re-raise; prefer structured logging and typed re-throws.


## 10. Configuration & Reproducibility

- Typed config models (Pydantic v2) for:
  - `AppConfig`, `LoggingConfig`, `TokenizerConfig`, `ModelConfig`, `TrainerConfig`, `DataConfig`, `ComputeConfig`.
- Config sources: `TOML` files + env var overrides.
- Every run writes a `manifest.json` including config, library versions, seed, CPU info, durations, artifacts.
- Seed all random sources (Python, NumPy, PyTorch, HF datasets where applicable).


## 11. Data Pipeline & Tokenizer Training

- Input: UTF‑8 text files under a corpus directory or a single combined file.
- Preprocessing: minimal normalization options in MVP (unicode normalization, lowercasing optional).
- Holdout: stratified-like split by file boundary or line boundary; deterministic via seed.
- Tokenizer training:
  - BPE via `tokenizers` with multi-threaded trainer; `vocab_size`, `min_frequency`.
  - SentencePiece optional (Unigram/BPE); use subprocess training with controlled threads.
- Validation metrics: coverage, OOV (if applicable), average tokens per char/word.
- Artifacts: tokenizer files + `tokenizer_manifest.json` with training stats.


## 12. Model Training (CPU-Only MVP)

- Backend: Hugging Face Transformers + PyTorch (CPU).
- Data: HF `datasets` with streaming or on-disk arrow, tokenized with selected tokenizer.
- Collation: causal LM collator with static or dynamic padding to `max_seq_len`.
- Trainer: Hugging Face `Trainer` with CPU settings.
- CPU threading:
  - `torch.set_num_threads(N)` and `torch.set_num_interop_threads(M)`.
  - `OMP_NUM_THREADS`, `MKL_NUM_THREADS` set via `ComputeProvider`.
  - DataLoader workers `num_workers` tuned to CPU cores.
- Metrics: loss, perplexity (eval), samples/sec (indicative), training time.
- Checkpointing: step/epoch intervals; best checkpoint by eval loss.


## 13. Storage & Artifacts

Directory layout (local):
```
corpus/              # user-provided corpus root (mounted read-only in containers)
artifacts/
  tokenizers/
    <tokenizer_id>/
      tokenizer.json | tokenizer.model
      manifest.json
      logs.jsonl
  models/
    <run_id>/
      manifest.json
      logs.jsonl
      eval/
        metrics.json
```

Notes:
- There is no separate runtime `runs/` directory; per‑run manifests and logs live under `artifacts/models/<run_id>/`.


## 14. API Surface (MVP)

- `POST /tokenizers/train` — start tokenizer training
- `GET /tokenizers/:id` — get tokenizer info/stats
- `POST /runs/train` — start model training
- `GET /runs/:id` — get run status/metrics
- `GET /runs/:id/logs` — tail per‑run logs
- `GET /runs/:id/logs/stream` — SSE stream of logs
- `GET /artifacts/:kind/:id` — download artifact
- `GET /healthz` — lightweight health
- `GET /readyz` — dependency readiness (e.g., Redis reachable)
- `POST /runs/:id/evaluate` — run evaluation on a split/dataset
- `GET /runs/:id/eval` — fetch evaluation results/summary

All request/response schemas are strictly typed Pydantic models. No `Any`.


## 15. UI (MVP)

- Pages:
  - Dashboard: recent runs, statuses.
  - New Tokenizer: select method + config + corpus + holdout.
  - New Training Run: select model + tokenizer + data + trainer config.
  - Run Details: live logs, metrics, checkpoints, artifacts.
- Implementation:
  - React + Vite + TypeScript (strict).
  - Typed API client with generated types from OpenAPI (FastAPI schema).


## 16. Testing & Quality Gates

- mypy: `--strict` with rules enforcing:
  - `disallow_any_* = True`, `no_implicit_optional = True`.
  - Disallow untyped defs and calls; require return types.
- pytest unit and integration tests for:
  - Tokenizer backends: training/encode/decode roundtrips on toy corpus.
  - Model backends: minimal steps to ensure loops execute and artifacts are produced.
  - Orchestrators: happy-path flows and error injection.
- Pre-commit: ruff, mypy, black (if used).


## 17. Concurrency & Performance Notes

- Tokenization: use library-native parallelism (threads) for training and encoding.
- Data pipeline: DataLoader `num_workers` tuned to CPU cores; chunked line reading.
- Avoid Python GIL hotspots by relying on native code paths (tokenizers, PyTorch, datasets).
- Limit memory footprint: streamed datasets, small batch sizes; gradient accumulation only if needed.


## 18. Security & Licensing

- Respect gated weights (e.g., LLaMA) and license checks; user supplies access tokens/agreements.
- Cache isolation and path validation for user-supplied corpus paths.


## 19. Roadmap

- MVP (local CPU):
  - BPE tokenizer (HF Tokenizers) + GPT‑2 small training end-to-end.
  - Strict types, DI container, centralized logging, error service.
  - UI to drive tokenizer + training runs and view logs/metrics.
- Next:
  - SentencePiece tokenizer backend.
  - LLaMA/Qwen adapters via Transformers with type-safe wrappers.
  - Resume/continue runs and artifact browser.
- Later:
  - Cloud compute provider (job submission + remote logs/artifacts).
  - Advanced metrics, curriculum options, early stopping/patience.
  - Plugin discovery via entry points for third-party backends.


## 20. Implementation Plan (Concrete Steps)

1. Scaffolding
   - Set up server package with mypy/ruff/pytest and FastAPI skeleton.
   - Implement LoggingService (JSON), ErrorService, ConfigService.
   - Implement DI container and registry patterns.
2. Contracts
   - Define tokenizer/model/data/compute protocols and config models.
3. Tokenizer MVP
   - HF Tokenizers BPE backend + orchestrator + API + tests.
4. Data Pipeline
   - Corpus provider + dataset builder; split/holdout and tokenization.
5. Model MVP
   - GPT‑2 backend via Transformers (CPU) + orchestrator + API + tests.
6. UI MVP
   - TS React app, strict mode; forms for tokenizer/training; live logs view.
7. Hardening
   - Type coverage review (no Any), test coverage for happy-path, run manifest.


## 21. Type Safety Guarantees

- Python: mypy strict across all modules; no `typing.Any`, no `typing.cast`.
  - For third-party libs lacking types, wrap in thin, typed adapters or create `.pyi` stubs.
- TypeScript: `strict: true`, `noImplicitAny: true`, `noUncheckedIndexedAccess: true`, `exactOptionalPropertyTypes: true`.


## 22. Success Criteria (MVP)

- Given a small cleaned corpus, user can:
  - Train a BPE tokenizer with a holdout and see coverage stats.
  - Train GPT‑2 small on CPU with visible progress and saved checkpoints.
  - View logs and metrics in UI; export tokenizer/model artifacts and a manifest.
- All code passes type checks and unit tests; no silent exceptions; consistent structured logs.

## 23. Containerization (Docker & Compose)

Services (compose profiles `dev` and `prod`):
- api: FastAPI + Uvicorn, CPU-only PyTorch, Poetry-managed runtime. Exposes port 8000.
- worker (recommended for multi-day async): same image as api, runs job consumer for training/eval.
- ui-dev (dev): Vite dev server with HMR; proxies `/api` → api; port 5173.
- ui (prod): Nginx serving built static assets.
- redis (required): durable queue and pub/sub for jobs/log streaming. Image `redis:7-alpine`.
- reverse-proxy (optional, prod): Traefik/Nginx routing `/` → ui, `/api` → api.
 - discord-status-bot (optional): `discord.py` bot that provides `/status`, `/subscribe`, and push alerts; queries API and listens to Redis Pub/Sub.

Base images:
- api/worker: `python:3.11-slim` with system deps; install via Poetry (no dev deps in prod).
- ui-dev build: `node:20-alpine`; ui-prod serve: `nginx:alpine`.

Volumes & caches:
- `./corpus:/data/corpus:ro`
- `./artifacts:/data/artifacts`
- `./runs:/data/runs`
- `./logs:/data/logs`
- Named volume `hf_cache:/hf-cache` with `HF_HOME=/hf-cache`

Env (examples):
- `APP_ENV=dev|prod`, `LOG_LEVEL=INFO`, `HF_HOME=/hf-cache`
- `REDIS_URL=redis://redis:6379/0` (dev/prod default) or Upstash URL for cloud
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `TOKENIZERS_PARALLELISM=1`
- Secrets via `.env` mounted (FastAPI loads with Pydantic Settings); never bake secrets into images.
 

Compose usage:
- `docker compose --profile dev up` → api + ui-dev + redis (+ worker)
- `docker compose --profile prod up` → api + ui-prod + redis (+ reverse-proxy/worker)

Resilience:
- Worker runs jobs out-of-process so API can restart without stopping training.
- Checkpoints and manifests on shared volume allow resume after restarts.

Windows note:
- Use Docker Desktop; bind mounts map project paths to containers. Commands are OS-agnostic via Compose.

## 24. Dependency & Tooling (Poetry, Ruff, Mypy)

Poetry:
- Use `pyproject.toml` for dependencies; commit `poetry.lock` for reproducibility.
- Dev dependencies: ruff, mypy, pytest, pre-commit.

Mypy (strict):
- Enforce no `Any` usage and no implicit optional.
- Representative config flags:
  - `disallow_any_generics = True`
  - `disallow_any_unimported = True`
  - `disallow_any_expr = True`
  - `disallow_any_decorated = True`
  - `disallow_any_explicit = True`
  - `no_implicit_optional = True`
  - `warn_redundant_casts = True`
  - `warn_unused_ignores = True`

Ruff:
- Enable `ANN401` to forbid `Any` in annotations.
- Ban `typing.cast` via banned-API rule to enforce “no casts”.
- Example (conceptual) `pyproject.toml` entries:

```
[tool.ruff]
target-version = "py311"

[tool.ruff.lint]
select = ["E","F","B","I","UP","ANN","PERF","S","TID","C90"]

[tool.ruff.lint.flake8-annotations]
allow-star-arg-any = false
mypy-init-return = true

[tool.ruff.lint.flake8-tidy-imports.banned-api]
"typing.cast" = { msg = "Do not use typing.cast; prefer precise types or adapters." }
```

Pre-commit:
- Hooks for ruff, mypy, pytest (fast subset); block merges on failures.

## 25. Config & Secrets (.env)

- Pydantic v2 Settings models for all configs (`AppConfig`, `LoggingConfig`, `TokenizerConfig`, `ModelConfig`, `TrainerConfig`, `DataConfig`, `ComputeConfig`).
- Sources: TOML files (checked-in defaults) + environment variables + `.env` for secrets.
- `.env.example` documents required keys (e.g., `HF_TOKEN` if needed for gated models).
- Secrets loaded only at runtime; do not commit `.env`.

## 26. Asynchronous & Long-Running Jobs

Goals:
- Training may run for days; must survive API/UI restarts and continue asynchronously.

Architecture:
- API enqueues jobs and immediately returns `run_id`.
- Worker consumes from Redis-backed queue (Redis required in all profiles).
- Queue implementation: RQ (Redis Queue) with `redis-py` client; our code depends on a typed JobQueue contract and a small adapter around RQ for strict typing.
- Worker writes logs to JSONL and periodic checkpoints; updates `manifest.json`.
- On startup, Worker reconciles and resumes incomplete runs from latest checkpoint.

Control & Observability:
- `GET /runs/:id` returns status and last heartbeat.
- `GET /runs/:id/logs` streams logs from the file (tail semantics).
- `POST /runs/:id/cancel` sets cancellation flag; worker checks between steps/batches.

Threading/Process model:
- Model training runs in the worker process (synchronous PyTorch loop).
- API remains responsive (async FastAPI) and performs non-blocking IO.

Notes on alternatives:
- Upstash vs redis-py: Upstash is a managed Redis service (deployment choice). We still use `redis-py` as the client. Set `REDIS_URL` to the Upstash URL (TLS) when using it.
- Celery: powerful but heavy for this MVP; adds brokers/results backends and more complexity. We can add a Celery adapter later if needed.
- arq/dramatiq: viable; arq is asyncio-native. We prefer RQ for simplicity and stability, wrapped behind our JobQueue contract for easy swapping.

Retry & failure semantics:
- Classify failures (`USER_INPUT`, `TRANSIENT`, `FATAL`).
- Default training policy: no auto-retry on `USER_INPUT`; bounded retries with backoff for `TRANSIENT`; no retry for `FATAL`.
- Persist failure classification and last exception in the run manifest; log with `error_code`.

Heartbeats & cancellation:
- Worker emits heartbeats to Redis key `runs:hb:<id>`.
- Status key: `runs:status:<id>`; cancellation flag: `runs:<id>:cancelled`.

## 27. Compose Profiles & Commands (Planned)

- Profiles:
  - `dev`: api, ui-dev, worker (optional), redis (optional).
  - `prod`: api, ui, reverse-proxy (optional), worker (optional), redis (optional).
- Common commands:
  - `docker compose --profile dev up -d`
  - `docker compose --profile dev logs -f api worker`
  - `docker compose --profile prod up -d`

## 28. Borrowed Patterns (Swarm/DiscordBot)

Adopted inspirations to improve reliability and maintainability:
- Logging context (DiscordBot): attach `request_id` and `instance_id` to every log record; dev-friendly console logs + JSONL per run.
- Typed DI container (DiscordBot/Swarm): explicit ServiceContainer for config/services; no global singletons.
- Settings via Pydantic (Swarm): nested env with `env_nested_delimiter="__"`, `.env` support, strict validation.
- Health endpoints (Swarm-inspired): `/healthz` and `/readyz` only (no Prometheus exporter).
- Queue durability (beyond DiscordBot BRPOP): use RQ for ack/retry/visibility; avoid LPUSH/BRPOP loss on crash.
 
- Structured contracts and protocols (both): standard interfaces for queues, tokenizers, models, compute providers.

Non-goals from these repos for MVP:
- Celery workers, autoscalers, and full observability stack (Prometheus/Grafana/Loki) — can be added later.
- Discord frontend — not required; UI remains web-based.

## 29. Discord Status Notifications (Optional, Personal DM-Only)

Purpose:
- Personal, DM-only status visibility through Discord without uploading data or controlling runs.

Capabilities:
- Commands (respond in DMs only; ignore guild contexts):
  - `/trainer runs` — list recent runs with IDs and statuses.
  - `/trainer status <run_id>` — fetch current status, last checkpoint, ETA (if available).
  - `/trainer watch <run_id>` — enable push updates for this run to your DMs.
  - `/trainer unwatch <run_id>` — stop updates.
- Push alerts (rate-limited and batched to DM):
  - Run queued, started, checkpoint saved, eval completed, run completed, run failed.
  - Periodic heartbeat summaries (e.g., every 5–10 minutes) with step/epoch progress.

Architecture:
- Run worker publishes JSON events to Redis Pub/Sub channel `runs:<run_id>:events`.
- Discord bot subscribes to Pub/Sub only for runs you “watch”.
- Personal mode storage (single user):
  - `discord:watching` (set of run_ids) indicates which runs to deliver to DM.
  - On startup, the bot opens/validates a DM with the owner and restores Pub/Sub subscriptions for run_ids in `discord:watching`.
- Status queries call the API (`GET /runs/:id`) for source-of-truth state. No corpus data is ever sent to Discord.

Safety & limits:
- Owner-only DMs: require `DISCORD_OWNER_ID`; ignore all events outside the owner’s DM.
- Redact sensitive paths in messages; never display corpus contents or file listings.
- Apply a per-DM rate limit and batch updates to avoid spam and Discord rate limits.

Containerization:
- Optional `discord-status-bot` container using `python:3.11-slim` + `discord.py`.
- Env: `DISCORD_TOKEN`, `DISCORD_OWNER_ID`, `REDIS_URL`, `API_BASE_URL`.

Testing:
- Unit test the event formatter, rate limiter, and subscription storage logic (fake Redis client).
- Integration test mocks API responses and simulates Pub/Sub events.

## 30. Model Evaluation & Testing

Goals:
- Provide quantitative and qualitative checks to validate trained models on CPU.

Evaluation pipeline:
- Trigger: automatically after training, or via `POST /runs/:id/evaluate`.
- Data: use the run’s validation split or a user-provided eval split/path.
- Metrics:
  - Validation loss and perplexity (causal LM cross-entropy).
  - Tokenization stats: average tokens per char/word, sequence length distribution.
  - Optional distinct-n and repetition indicators on generated samples.
- Qualitative: deterministic sample generations from a fixed prompt set (seeded), saved to `samples.jsonl`.

Artifacts:
- `artifacts/models/<run_id>/eval/metrics.json`
- `artifacts/models/<run_id>/eval/samples.jsonl`
- Update run `manifest.json` with eval summary (loss, ppl, dataset info, seed).

Acceptance guidance (CPU‑first, tiny models):
- Sanity target: validation perplexity decreases vs. the first epoch.
- For regression baselines, compare to a previous run’s metrics; flag regressions in the UI.

API:
- `POST /runs/:id/evaluate` — request eval on a specific split or external path.
- `GET /runs/:id/eval` — return latest evaluation summary and artifact pointers.

Testing strategy:
- Tokenizers: encode/decode roundtrip on toy samples; coverage stats on a small corpus.
- Training loop: run a tiny training step on a toy corpus; assert loss decreases between first two evals.
- Evaluation: compute perplexity on a known toy dataset; verify deterministic results with fixed seeds.
- Orchestrators: happy-path and error injection verifying error codes and no silent failures.

## 31. Security & Correlation (Implemented)

- Request correlation: middleware sets and echoes `X-Request-ID` for every response.
- Error bodies: all errors include `code`, `message`, and `request_id` (see `core/errors/handlers.py`).
- API key auth: if configured, requests must include `X-Api-Key`; invalid/missing key returns HTTP 401 with code `UNAUTHORIZED`.

## 32. Training Events & Discord Integration (Implemented)

- Worker publishes JSON events for training lifecycle to Redis channel `trainer:events`.
- Event schema: `trainer.train.started.v1`, `trainer.train.progress.v1`, `trainer.train.completed.v1`, `trainer.train.failed.v1` (see `events/trainer.py`).
- `request_id` equals `run_id` for direct correlation; payload contains `user_id` for Discord DM routing.
- DiscordBot subscribes and sends rich embeds to users.

## 33. Run Status Messages (Implemented)

- Status keys: `runs:status:<run_id>` and heartbeats `runs:hb:<run_id>`.
- A human-readable message is stored at `runs:msg:<run_id>` on cancellation/failure/completion and exposed via `GET /runs/{id}` as `message`.

## 34. Railway Deployment Notes

- Services: API (uvicorn) and worker (rq), plus Redis addon.
- Ensure start commands execute in `server/` (per-service root or `cd server && ...`).
- Required env: `REDIS_URL`, `SECURITY__API_KEY` (if enforcing auth), and artifacts/log roots.
