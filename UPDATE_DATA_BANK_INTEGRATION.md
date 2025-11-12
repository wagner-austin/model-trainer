# Data‑Bank API Integration Update (Strict, Typed, Tested)

This document describes the changes to integrate model‑trainer with data‑bank‑api while maintaining strict type safety, zero drift, and full test coverage.

## Summary
- Add first‑class support for downloading corpora from data‑bank‑api using server‑generated `file_id` (sha256).
- Keep existing `corpus_path` flows intact; support `corpus_file_id` as an alternative source (exactly one must be provided).
- Enforce strict typing across all changes (no `Any`, no casts, no `type: ignore`).
- Add resilient, resumable download service with integrity verification and request timeouts.
- Extend tests to 100% statements and branches; include negative/error paths.

## Required Changes

### Settings
Update `server/model_trainer/core/config/settings.py` (`AppConfig`) with:
- `data_bank_api_url: str = ""`
- `data_bank_api_key: str = ""`
These are populated via env/TOML as with existing config.

### API Schema
Update `server/model_trainer/api/schemas/runs.py` (`TrainRequest`):
- Add `corpus_file_id: Annotated[str | None, Field(default=None, description="data-bank-api file ID")]`.
- Validator to enforce XOR: exactly one of `corpus_path` or `corpus_file_id` must be provided.

### Data Service
Add `server/model_trainer/core/services/data/corpus_fetcher.py`:
- `CorpusFetcher(api_url: str, api_key: str, cache_dir: Path)`
- `fetch(file_id: str) -> Path`
  - HEAD for `Content-Length` and `ETag`.
  - Resumable GET with `Range` (1 MiB chunks), `timeout=600s`.
  - Verify downloaded size (and optionally sha256 vs ETag if desired).
  - Atomic rename `*.tmp -> *.txt` in cache.

### Orchestrator
Update `server/model_trainer/orchestrators/training_orchestrator.py`:
- When `request.corpus_file_id` is set, resolve to local cache path via `CorpusFetcher`. Otherwise use provided `corpus_path`.
- Pass resolved `corpus_path` into `TrainJobPayload`.

### Environment
Set these for API and Worker on Railway:
- `DATA_BANK_API_URL="http://data-bank-api.railway.internal"`
- `DATA_BANK_API_KEY="<trainer-key>"`

## Typing, Linting, Guards
- mypy `--strict` must pass (no `Any`, no casts, no ignores).
- Ruff must pass; `typing.cast` is banned.
- Existing guard script must remain green (no drift markers, no suppressed checks).

## Tests (100% Statements + Branches)
Add unit tests using `httpx.MockTransport` to cover:
- HEAD/GET success, resume from partial file, final integrity verify.
- 401/403/404/416/5xx mappings, transport errors with retries.
- Already complete (short‑circuit) with/without integrity verification.

Extend API tests (`runs` endpoints):
- Train request with `corpus_file_id` triggers fetcher call; payload includes resolved path.
- Error cases bubble back as 400 with details when fetch fails.

## Make Commands
- `make check` (root): runs guards + lint + tests with branch coverage.
- `make check` (server/): runs mypy/ruff/pytest in Poetry env; ensure 100%.

## Migration/Notes
- Existing `corpus_path` behavior remains unchanged.
- `corpus_file_id` becomes the recommended path for large corpora via data‑bank‑api.
- Use the internal Railway URL and X‑API‑Key.

## Risks & Mitigations
- Large downloads: resumable transfers, long timeouts, and integrity checks reduce flakiness.
- Storage pressure: rely on data‑bank‑api free‑space guards and eventual TTLs.
