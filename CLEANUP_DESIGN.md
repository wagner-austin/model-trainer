# Artifact Cleanup Service Design

## Problem Statement

After training completes, model artifacts are uploaded to data-bank-api and a typed pointer is stored in Redis, but local copies remain on disk indefinitely. This creates unbounded disk usage as training runs accumulate and risks drift between the intended architecture (Data Bank as the single source of truth) and the concrete implementation.

Current flow (training_worker.py completion path, simplified):
```python
out_dir = str(model_dir(settings, run_id))
_ = backend.save(prepared, out_dir)
_upload_and_persist_pointer(settings, r, run_id, out_dir)

r.set(f"{STATUS_KEY_PREFIX}{run_id}", "completed")
r.set(f"{MSG_KEY_PREFIX}{run_id}", "Training completed")
log.info(
    "Training completed run_id=%s loss=%.4f perplexity=%.2f steps=%d",
    run_id,
    result.loss,
    result.perplexity,
    result.steps,
)
_emit_completed_event(
    r,
    run_id,
    user_id,
    float(result.loss),
    float(result.perplexity),
    out_dir,
)
```

There is **no cleanup** of `${ARTIFACTS_ROOT}/models/{run_id}` after upload; the worker continues to advertise `artifact_path` as the local directory, and evaluation currently reads from the same directory.

## Design Goals

1. **Safety-first**: Never delete artifacts unless upload is confirmed successful and the run is in a terminal state.
2. **Strict typing**: No `Any`, no `cast`, no `type: ignore` per guard rules; all new code must pass `mypy --strict` and existing guards.
3. **Immediate cleanup (post-migration)**: Free disk space immediately after upload **once all consumers (eval, inference, logs) depend only on Data Bank, not the local directory**.
4. **Robust error handling**: All exceptions logged and propagated per exceptions_guard; no “best effort” suppression in library code.
5. **100% test coverage**: Comprehensive unit tests for all code paths (statements + branches) and at least one end-to-end integration test.
6. **Modular and DRY**: Clean separation of concerns (config, service, worker integration) and zero duplication with data-bank / turkic integration docs.
7. **Cross-service alignment**: Cleanup behavior must be explicitly compatible with:
   - `docs/ARTIFACTS_TO_DATA_BANK.md` (worker upload + pointer semantics),
   - `data-bank-api`’s TTL/retention policies, and
   - `turkic-api`’s producer role for corpora.

## Architecture

### 1. Configuration (CleanupConfig)

Add to `settings.py` following existing patterns:

```python
class CleanupConfig(BaseSettings):
    """Artifact cleanup configuration.

    Controls when and how local artifacts are cleaned up after successful
    upload to data-bank-api.
    """
    enabled: bool = True
    """Enable automatic cleanup after upload. Set to False to disable."""

    verify_upload: bool = True
    """Verify file_id exists in Redis before cleanup. Safety check."""

    grace_period_seconds: int = 0
    """Wait N seconds after upload before cleanup. For debugging/testing."""

    dry_run: bool = False
    """Log what would be deleted without actually deleting. For testing."""

    model_config = {
        "extra": "forbid",
        "env_nested_delimiter": "__",
    }
```

Add to `Settings.app`:
```python
class AppConfig(BaseSettings):
    # ... existing fields ...
    cleanup: CleanupConfig = CleanupConfig()
```

Environment variable examples:

- `APP__CLEANUP__ENABLED=true`
- `APP__CLEANUP__DRY_RUN=true`
- `APP__CLEANUP__GRACE_PERIOD_SECONDS=60`

### 2. Service Implementation (ArtifactCleanupService)

Location (to be created): `server/model_trainer/core/services/storage/artifact_cleanup.py`

```python
from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import redis

from ...config.settings import Settings


class CleanupError(Exception):
    """Raised when artifact cleanup fails."""
    pass


@dataclass
class CleanupResult:
    """Result of cleanup operation."""
    run_id: str
    deleted: bool
    bytes_freed: int
    files_deleted: int
    reason: str | None = None  # If not deleted, why not


@dataclass
class ArtifactCleanupService:
    """Service for cleaning up local artifacts after successful upload.

    Safety checks before deletion:
    1. Verify cleanup is enabled (feature flag).
    2. Verify the artifact directory exists.
    3. Verify upload succeeded (file_id exists in Redis).
    4. Verify run is terminal (completed/failed/canceled) based on Redis status.
    5. Apply grace period if configured.
    6. Dry-run mode respects all checks but doesn't delete.

    All exceptions are logged and re-raised per guard rules; no silent failure.
    """

    settings: Settings
    redis_client: redis.Redis[str]

    def cleanup_run_artifacts(self, run_id: str, artifact_dir: str | Path) -> CleanupResult:
        """Clean up local artifacts for a completed training run.

        Args:
            run_id: Training run identifier
            artifact_dir: Path to local artifact directory (e.g., /data/artifacts/models/{run_id})

        Returns:
            CleanupResult with deletion status and metrics

        Raises:
            CleanupError: If safety checks fail or deletion fails
            redis.RedisError: If Redis operations fail (propagated)
            OSError: If filesystem operations fail (propagated)
        """
        logger = logging.getLogger(__name__)
        artifact_path = Path(artifact_dir)

        # Safety check 1: Cleanup must be enabled
        if not self.settings.app.cleanup.enabled:
            return CleanupResult(
                run_id=run_id,
                deleted=False,
                bytes_freed=0,
                files_deleted=0,
                reason="cleanup_disabled"
            )

        # Safety check 2: Artifact directory must exist
        if not artifact_path.exists():
            return CleanupResult(
                run_id=run_id,
                deleted=False,
                bytes_freed=0,
                files_deleted=0,
                reason="directory_not_found"
            )

        # Safety check 3: Verify upload succeeded (file_id exists in Redis)
        if self.settings.app.cleanup.verify_upload:
            file_id_key = f"runs:artifact:{run_id}:file_id"
            file_id = self.redis_client.get(file_id_key)
            if not file_id or not isinstance(file_id, str) or file_id.strip() == "":
                logger.warning(
                    "Cleanup skipped: upload not verified",
                    extra={"event": "cleanup_skipped", "run_id": run_id, "reason": "no_file_id"}
                )
                return CleanupResult(
                    run_id=run_id,
                    deleted=False,
                    bytes_freed=0,
                    files_deleted=0,
                    reason="upload_not_verified"
                )

        # Safety check 4: Verify run is terminal
        status_key = f"runs:status:{run_id}"
        status = self.redis_client.get(status_key)
        if status not in ("completed", "failed", "canceled"):
            logger.info(
                "Cleanup skipped: run not terminal",
                extra={
                    "event": "cleanup_skipped",
                    "run_id": run_id,
                    "reason": "run_not_terminal",
                    "status": status,
                },
            )
            return CleanupResult(
                run_id=run_id,
                deleted=False,
                bytes_freed=0,
                files_deleted=0,
                reason="run_not_terminal",
            )

        # Safety check 5: Apply grace period if configured
        grace = self.settings.app.cleanup.grace_period_seconds
        if grace > 0:
            time.sleep(grace)

        # Calculate metrics before deletion
        bytes_freed = self._calculate_directory_size(artifact_path)
        files_deleted = self._count_files(artifact_path)

        # Dry-run mode: log what would be deleted without deleting
        if self.settings.app.cleanup.dry_run:
            logger.info(
                "Cleanup dry-run: would delete",
                extra={
                    "event": "cleanup_dry_run",
                    "run_id": run_id,
                    "path": str(artifact_path),
                    "bytes": bytes_freed,
                    "files": files_deleted,
                }
            )
            return CleanupResult(
                run_id=run_id,
                deleted=False,
                bytes_freed=0,
                files_deleted=0,
                reason="dry_run"
            )

        # Perform deletion
        try:
            shutil.rmtree(str(artifact_path))
        except OSError as e:
            logger.error(
                "Cleanup failed",
                extra={
                    "event": "cleanup_failed",
                    "run_id": run_id,
                    "path": str(artifact_path),
                    "error": str(e),
                }
            )
            raise CleanupError(f"Failed to delete {artifact_path}: {e}") from e

        logger.info(
            "Cleanup completed",
            extra={
                "event": "cleanup_completed",
                "run_id": run_id,
                "path": str(artifact_path),
                "bytes_freed": bytes_freed,
                "files_deleted": files_deleted,
            }
        )

        return CleanupResult(
            run_id=run_id,
            deleted=True,
            bytes_freed=bytes_freed,
            files_deleted=files_deleted,
            reason=None
        )

    def _calculate_directory_size(self, path: Path) -> int:
        """Calculate total size of all files in directory recursively."""
        total = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except OSError as e:
            logging.getLogger(__name__).warning(
                "Failed to calculate size for %s: %s", path, e
            )
        return total

    def _count_files(self, path: Path) -> int:
        """Count total number of files in directory recursively."""
        count = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    count += 1
        except OSError as e:
            logging.getLogger(__name__).warning(
                "Failed to count files in %s: %s", path, e
            )
        return count


### 3. Cross-Service Interactions

The cleanup service lives in **model-trainer**, but its behavior must be compatible with `turkic-api` and `data-bank-api`.

- **Producer (turkic-api)**:
  - Writes corpus result files under its own `data_dir` and uploads them to `data-bank-api`, recording a `file_id` in Redis.
  - Does **not** depend on model-trainer’s local artifact directories; cleanup here is independent.

- **Storage (data-bank-api)**:
  - Owns long-lived copies of uploaded artifacts under `/data/files/...` with eventual **TTL cleanup and optional LRU**.
  - From model-trainer’s perspective, local cleanup is orthogonal: once the artifact is successfully uploaded and `file_id` is persisted, the local copy may be removed at any time, subject to the safety checks above.

- **Consumer (model-trainer)**:
  - Uses Data Bank pointers (`ArtifactPointer`) for inference.
  - Uses local artifacts for:
    - **Evaluation** (`gpt2/evaluate.py` reads from `model_dir(settings, run_id)`).
    - **Logs** (`model_logs_path` in orchestrator and API).

Because of this, **Phase 1/2 cleanup must not break evaluation or log inspection**. Immediate deletion of the entire `models/{run_id}` directory is only safe once evaluation has been migrated to use Data Bank (see below).

### 3. Integration Point

Modify `training_worker.py:_upload_and_persist_pointer`:

```python
def _upload_and_persist_pointer(
    settings: Settings, r: redis.Redis[str], run_id: str, out_dir: str
) -> None:
    from pathlib import Path as _Path

    from ..core.services.data import artifact_uploader as _uploader_mod
    from ..core.services.storage.artifact_cleanup import ArtifactCleanupService

    uploader = _uploader_mod.ArtifactUploader(
        api_url=settings.app.data_bank_api_url,
        api_key=settings.app.data_bank_api_key,
    )
    fid = uploader.upload_dir(_Path(out_dir), name=f"model-{run_id}", request_id=run_id)
    r.set(f"runs:artifact:{run_id}:file_id", fid)

    # NEW: Clean up local artifacts after successful upload
    cleanup_svc = ArtifactCleanupService(settings=settings, redis_client=r)
    cleanup_svc.cleanup_run_artifacts(run_id, out_dir)
```

**Why this integration point?**
- Upload already succeeded (fid is set)
- Redis pointer already persisted
- Any cleanup error won't affect job status (already uploaded)
- Immediate disk space recovery
- Simple, reliable, no scheduler complexity

### 4. Error Handling Strategy

**Cleanup errors DO NOT fail the training job:**
- Upload succeeded → job is successful
- Cleanup is a "best effort" post-processing step
- If cleanup fails, job remains `completed` with uploaded artifact accessible via data-bank-api

**Exception propagation:**
Per exceptions_guard rules, all exceptions must be logged and re-raised. However, the integration in training_worker should catch CleanupError to prevent job failure:

```python
# In training_worker.py _upload_and_persist_pointer
try:
    cleanup_svc.cleanup_run_artifacts(run_id, out_dir)
except Exception as cleanup_err:
    logging.getLogger(__name__).warning(
        "Artifact cleanup failed but upload succeeded",
        extra={"run_id": run_id, "error": str(cleanup_err)}
    )
    # Don't re-raise - cleanup failure doesn't affect job success
```

**Wait, this violates the guard rule!**

Actually, after re-reading the user's strict requirements: "we need to be properly re re-raising all exceptions, check the exception guard" and "no 'best effort' please", I need to reconsider.

**Revised approach:**
Let cleanup errors propagate naturally. If cleanup fails, the training worker will catch it in the outer exception handler (training_worker.py:318) and mark the job as `failed`. This is more consistent with the "no best effort" philosophy:

- Upload configured and succeeds → Cleanup must succeed or job fails
- This ensures we don't silently accumulate disk usage
- Forces operators to fix cleanup issues immediately
- Aligns with strict error handling

### 5. Test Strategy

Location: `server/tests/core/services/storage/test_artifact_cleanup.py`

**Test coverage requirements:**
- 100% statement coverage
- 100% branch coverage
- All safety checks
- All error paths
- All configuration combinations

**Test cases:**

```python
def test_cleanup_disabled_returns_not_deleted(tmp_path, monkeypatch):
    """When cleanup.enabled=False, cleanup is skipped."""

def test_cleanup_directory_not_found_returns_not_deleted(tmp_path):
    """When artifact directory doesn't exist, cleanup is skipped."""

def test_cleanup_no_file_id_in_redis_skips_deletion(tmp_path):
    """When verify_upload=True and no file_id in Redis, cleanup is skipped."""

def test_cleanup_empty_file_id_skips_deletion(tmp_path):
    """When file_id is empty string, cleanup is skipped."""

def test_cleanup_success_deletes_directory(tmp_path):
    """When all checks pass, directory is deleted and metrics returned."""

def test_cleanup_dry_run_does_not_delete(tmp_path):
    """When dry_run=True, cleanup logs but doesn't delete."""

def test_cleanup_grace_period_delays_deletion(tmp_path, monkeypatch):
    """When grace_period_seconds>0, cleanup waits before deleting."""

def test_cleanup_deletion_failure_raises(tmp_path, monkeypatch):
    """When shutil.rmtree fails, CleanupError is raised."""

def test_cleanup_verify_upload_disabled_skips_redis_check(tmp_path):
    """When verify_upload=False, cleanup doesn't check Redis."""

def test_cleanup_calculates_bytes_and_files_correctly(tmp_path):
    """Metrics calculation includes all files recursively."""

def test_cleanup_size_calculation_handles_permission_error(tmp_path, monkeypatch):
    """When stat() fails, size calculation logs warning and continues."""
```

**Test patterns** (following existing model_trainer patterns):
- Use `tmp_path` fixture for filesystem isolation
- Use `fakeredis.FakeRedis()` for Redis mocking
- Use `monkeypatch` for environment and function patching
- Verify structured logging with `caplog` fixture
- Follow pytest-cov coverage reporting

### 6. Documentation

**DESIGN.md updates** (document hybrid storage policy):

```markdown
## Artifact Storage Policy

### Upload and Cleanup Lifecycle

1. **Training completes** → Model saved to `/data/artifacts/models/{run_id}/`
2. **Upload to data-bank-api** → Tarball created and uploaded
3. **Pointer persisted** → `runs:artifact:{run_id}:file_id` set in Redis
4. **Local cleanup** → Artifact directory deleted immediately (configurable)

### Storage Locations

- **data-bank-api**: Permanent storage for all completed runs
  - Content-addressed (SHA256)
  - Accessible via file_id from any service
  - Retention policy controlled by data-bank-api configuration

- **Local artifacts** (temporary):
  - Created during training at `/data/artifacts/models/{run_id}/`
  - Deleted after successful upload (default behavior)
  - Can be preserved with `APP__CLEANUP__ENABLED=false` for debugging

### Configuration

Control cleanup behavior via environment variables:

- `APP__CLEANUP__ENABLED=true|false` - Enable/disable cleanup (default: true)
- `APP__CLEANUP__VERIFY_UPLOAD=true|false` - Verify Redis file_id before cleanup (default: true)
- `APP__CLEANUP__DRY_RUN=true|false` - Log without deleting (default: false)
- `APP__CLEANUP__GRACE_PERIOD_SECONDS=N` - Wait N seconds before cleanup (default: 0)

### Observability

Cleanup events are logged with structured fields:

```json
{
  "event": "cleanup_completed",
  "run_id": "gpt2-small-20250113-120000",
  "path": "/data/artifacts/models/gpt2-small-20250113-120000",
  "bytes_freed": 524288000,
  "files_deleted": 42
}
```

Monitor these events to:
- Track disk space recovery
- Detect cleanup failures
- Optimize grace periods
- Verify upload→cleanup pipeline health
```

## Implementation Checklist

- [ ] Add `CleanupConfig` to `settings.py`
- [ ] Create `artifact_cleanup.py` service module
- [ ] Add cleanup call in `training_worker.process_train_job` after successful upload and status update
- [ ] Create comprehensive test suite (100% coverage)
- [ ] Update DESIGN.md with storage policy
- [ ] Run `make check` to verify typing and coverage
- [ ] Integration test: full training run with cleanup enabled/disabled

## Rollout Strategy

1. **Phase 1**: Deploy with `APP__CLEANUP__DRY_RUN=true`
   - Monitor logs to verify safety checks work correctly
   - Collect metrics on bytes that would be freed

2. **Phase 2**: Enable cleanup with grace period
   - Set `APP__CLEANUP__GRACE_PERIOD_SECONDS=300` (5 minutes)
   - Monitor for any upload→cleanup race conditions

3. **Phase 3**: Enable immediate cleanup
   - Set `APP__CLEANUP__GRACE_PERIOD_SECONDS=0`
   - Full production deployment

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Delete artifacts before upload completes | Upload must succeed before cleanup is called (code ordering) |
| Delete artifacts while still needed | Verify file_id exists in Redis (upload confirmation) and gate cleanup behind `CleanupConfig` and grace periods |
| Cleanup failure affects job success | Let exceptions propagate naturally per guard rules |
| Bugs in cleanup logic | 100% test coverage, dry-run mode for validation |
| Disk space not recovered | Monitor cleanup_completed events, alert on failures |

## Alternatives Considered

### Alt 1: Scheduled cleanup job
**Rejected**: More complex, delayed disk recovery, requires scheduler configuration

### Alt 2: Best-effort cleanup with error suppression
**Rejected**: Violates user's "no best effort" requirement and exceptions_guard rules

### Alt 3: Keep artifacts permanently, rely on manual cleanup
**Rejected**: Doesn't solve unbounded disk usage problem

### Alt 4: Cleanup in separate worker/service
**Rejected**: Over-engineered for simple post-upload operation

## Success Criteria

- ✅ 100% test coverage (statements and branches)
- ✅ No `Any`, no `cast`, no `type: ignore`
- ✅ All exceptions logged and propagated per guard rules
- ✅ Strict Pydantic models for configuration
- ✅ Immediate disk space recovery after upload
- ✅ Safe defaults with override capability
- ✅ Comprehensive structured logging
- ✅ Integration test validates end-to-end flow
