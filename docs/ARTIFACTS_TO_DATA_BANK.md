# Centralize Training Artifacts in Data Bank

## Overview

In multi‑service deployments (e.g., Railway), the Model‑Trainer API and Worker run in separate services with isolated volumes. Today the Worker writes model artifacts to its local filesystem under `Settings.app.artifacts_root` (default `/data/artifacts`). The API’s artifact routes read from the API service’s volume, which does not contain the Worker’s files, and inference services cannot reliably access models.

This document proposes and specifies a complete migration to storing model/tokenizer artifacts in the Data Bank service. Workers upload trained artifacts to Data Bank and record a typed pointer (file_id). The API exposes pointer endpoints. Inference downloads by pointer from Data Bank. There are no fallbacks or back‑compat paths.

## Goals

- Single source of truth for artifacts in Data Bank.
- Strict typing; no `Any`, no `casts`, no `type: ignore`.
- DRY, modular responsibilities:
  - Orchestrators: enqueue only
  - Workers: train + upload
  - API: surface pointers
  - Inference: download + serve
- Maintain `make check` (ruff, mypy --strict, guards) and 100% statement/branch coverage.

## Non‑Goals

- Supporting legacy queue payloads (`corpus_path`) or legacy local artifact serving in API.
- Building the inference service in this change (documented as consumer).

## Current Behavior (for context)

- Artifacts written under: `${ARTIFACTS_ROOT}/models/{run_id}` (and `${ARTIFACTS_ROOT}/tokenizers/{tokenizer_id}` for tokenizers).
- Per‑run `manifest.json` and `logs.jsonl` live in the run directory.
- API artifact routes read from the API service volume; no cross‑service visibility.

## Proposed Architecture

1. After successful training, the Worker archives the run directory and uploads it to Data Bank.
2. On success, the Worker stores a typed pointer (Data Bank `file_id`) in Redis and updates the run `manifest.json`.
3. The API exposes pointer endpoints:
   - `GET /runs/{run_id}/artifact` → `{ "storage": "data-bank", "file_id": "..." }`
   - `GET /tokenizers/{tokenizer_id}/artifact` → same shape (optional but symmetric).
4. Inference service uses the pointer to download and load the model.

## Data Flow (end‑to‑end)

- Request: API receives `corpus_file_id`, orchestrator enqueues fid (no path resolution). (Already implemented.)
- Worker: resolves `corpus_file_id` via `CorpusFetcher` to local cache, trains, writes `save_pretrained(out_dir)`.
- Upload: Worker streams a tar of `out_dir` to Data Bank `/files` with `X-API-Key`.
- Pointer: Worker saves `file_id` → Redis and adds `artifact_pointer` to `manifest.json`.
- Consumption: API returns pointer; inference downloads tar, extracts, and loads the model.

## Interfaces and Types

### Artifact Pointer

```python
class ArtifactPointer(BaseModel):
    storage: Literal["data-bank"]
    file_id: str

    model_config = {"extra": "forbid", "validate_assignment": True}
```

### Redis Keys

- `runs:artifact:{run_id}:file_id = <str>`
- `tokenizer:{tokenizer_id}:file_id = <str>` (optional)

### API Pointer Endpoints

- `GET /runs/{run_id}/artifact` → `200` with `ArtifactPointer`, otherwise `404`.
- `GET /tokenizers/{tokenizer_id}/artifact` (optional) → same.

## Worker Changes

- Add `ArtifactUploader` helper (pure, typed):
  - `upload_dir(path: str, *, name: str) -> str` returns `file_id`.
  - Uses streamed tar creation, `httpx` POST to `${DATA_BANK_API_URL}/files`, header `X-API-Key`.
- Training Worker completion path:
  - On success:
    - Create tar from `${ARTIFACTS_ROOT}/models/{run_id}` (or stream directory).
    - Upload; on 2xx with valid JSON containing `file_id` → persist pointer to Redis and into `manifest.json`.
  - On failure (HTTP error / invalid JSON / missing `file_id`):
    - Mark job `failed`; message: `artifact upload failed`.
    - No local‑only fallback.
- Tokenizer Worker (optional): mirror the above under `${ARTIFACTS_ROOT}/tokenizers/{tokenizer_id}`.

### Required Worker Env

- `APP__DATA_BANK_API_URL`
- `APP__DATA_BANK_API_KEY`

## API Changes

- Add pointer endpoints (no file proxying):
  - `GET /runs/{run_id}/artifact` reads Redis key and returns `ArtifactPointer`.
  - `GET /tokenizers/{tokenizer_id}/artifact` (optional) reads Redis and returns pointer.
- Error handling: return `404` if no pointer yet.

## Logging and Observability

- Worker starts RQ programmatically (no `exec()`), so configured console logging remains active.
- Console JSON formatter includes extras: `category`, `service`, `event`, `run_id`, `tokenizer_id`, `error_code`.
- Worker logs `artifact_uploaded` with `file_id` and `run_id`.

## Testing Plan

- Unit (Uploader):
  - Success (2xx with `file_id`), 4xx/5xx, malformed JSON, missing `file_id`.
  - Ensure no `Any`/casts/ignores; mypy --strict passes.
- Worker integration (no network):
  - Monkeypatch `httpx.post` to return stubbed responses.
  - Verify upload invoked, Redis pointer set, manifest updated (success).
  - Verify failure sets job to failed with message `artifact upload failed`.
- API pointers:
  - 200 when present, 404 when absent.
- Coverage:
  - 100% statement and branch coverage for updated files.

## Deployment Plan

- No back‑compat. Steps:
  1. Drain RQ queues (remove old jobs enqueued under legacy payloads).
  2. Deploy Worker with Data Bank env configured.
  3. Deploy API with pointer endpoints.
  4. Submit a training job; confirm pointer creation and `file_id` presence.
  5. Build/point inference service to consume pointers.

## Risks and Mitigations

- Large upload sizes → timeouts: set generous timeout, chunked streaming, and clear error surface.
- Data storage costs → retention policy in Data Bank; scheduled cleanup of older runs if needed.
- Egress for inference → cache extracted model locally per run; validate checksum if desired.

## Future Work

- Stream per‑run logs to Redis lists/streams; API streams SSE from Redis.
- Signed upload URLs (reduce key scope/rotation burden).
- Artifact signing and checksum verification on download.

