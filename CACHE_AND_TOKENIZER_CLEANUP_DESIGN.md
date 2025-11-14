# Corpus Cache & Tokenizer Cleanup Design

## 1. Problem Statement

The current artifact cleanup feature handles **model training artifacts** under `artifacts_root/models/{run_id}`. However, two other storage areas can grow without bound:

- **Corpus cache**: worker-local copies of corpora fetched from data-bank-api (e.g., `/data/corpus_cache/{file_id}.txt`).
- **Tokenizers**: tokenizer artifacts under `artifacts_root/tokenizers/{tokenizer_id}`.

If left unmanaged, both can cause unbounded disk growth over time, especially in long-running deployments with many corpora and tokenizer configurations.

Constraints from the existing system and design culture:

- Strict typing; no `Any`, no casts, no `type: ignore`.
- No "best-effort" behavior; failures must be explicit and visible.
- No back-compat behavior that silently re-enables local-only flows.
- Centralization in data-bank-api as the source of truth for artifacts.
- 100% statements + branches coverage for cleanup logic.

This document specifies **separate, explicit cleanup policies** for corpus cache and tokenizers that respect these constraints and do not reintroduce local fallbacks.

---

## 2. Goals & Non-Goals

### 2.1 Goals

1. **Bounded corpus cache**: ensure `/data/corpus_cache` cannot grow without limit.
2. **Safe tokenizer cleanup**: allow removal of tokenizer artifacts that are no longer referenced, without breaking any active or historical runs.
3. **Strict error handling**: any cleanup failure is a hard failure (for the cleanup operation), not silently ignored.
4. **No behavioral regressions**: do not reintroduce local-only serving paths once Data Bank integration is in place.
5. **Design symmetry**: reuse patterns from model artifact cleanup (typed config, service modules, unit + integration tests).

### 2.2 Non-Goals (Initial)

- Implementing a background scheduler or cron service. Initial integration will be via explicit commands or maintenance endpoints.
- Auto-detecting "unused" tokenizers by static analysis of all downstream services. We limit the scope to **Model-Trainer's own references** (e.g., manifests, active configs).
- Implementing cross-service retention policies (e.g., automatically deleting Data Bank files). Data Bank remains the long-term archive owner.

---

## 3. Corpus Cache Cleanup

### 3.1 Semantics

Corpus cache entries are **pure cache**:

- Canonical source: data-bank-api (files service).
- Local path: `Settings.app.data_root / "corpus_cache" / "{file_id}.txt"` (existing behavior).
- Safety: any cache file may be deleted at any time; the worker will re-download via `CorpusFetcher` when needed.

There is no back-compat or correctness requirement to retain these files after use, beyond performance considerations.

### 3.2 Configuration (`CorpusCacheCleanupConfig`)

Add a dedicated config nested under `Settings`:

```python
class CorpusCacheCleanupConfig(BaseSettings):
    enabled: bool = False
    max_bytes: int = 10 * 1024 * 1024 * 1024  # 10 GiB
    min_free_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GiB
    eviction_policy: Literal["lru", "oldest"] = "lru"

    model_config = {
        "extra": "forbid",
        "env_nested_delimiter": "__",
    }
```

Integrate into `Settings` as:

```python
class AppConfig(BaseSettings):
    # existing fields...
    corpus_cache_cleanup: CorpusCacheCleanupConfig = CorpusCacheCleanupConfig()
```

Environment examples:

- `APP__CORPUS_CACHE_CLEANUP__ENABLED=true`
- `APP__CORPUS_CACHE_CLEANUP__MAX_BYTES=10737418240`
- `APP__CORPUS_CACHE_CLEANUP__MIN_FREE_BYTES=2147483648`

### 3.3 Service (`CorpusCacheCleanupService`)

Location: `server/model_trainer/core/services/data/corpus_cache_cleanup.py`

Responsibilities:

- Inspect the corpus cache directory (`Settings.app.data_root / "corpus_cache"`).
- Compute total size and free space on the underlying filesystem.
- When `enabled=True` and either:
  - total cache size exceeds `max_bytes`, or
  - free space is below `min_free_bytes`,
  perform eviction according to `eviction_policy`.

Eviction algorithm (LRU):

- For each file in the cache directory:
  - Collect `(path, size, last_access_ts)`. Use `stat().st_atime` for access time; if unavailable, fall back to `st_mtime`.
- Sort by ascending `last_access_ts` (least recently used first).
- Delete files in that order, updating the running totals, until both:
  - `total_cache_size <= max_bytes`, and
  - `free_space >= min_free_bytes`.

Error handling (strict, no best-effort):

- All filesystem errors (`OSError`) during scanning or deletion **raise** a typed `CorpusCacheCleanupError`.
- The cleanup operation has no silent paths: either it completes successfully (with a deterministic `CleanupResult`), or it fails loudly.

Result type:

```python
@dataclass
class CorpusCacheCleanupResult:
    deleted_files: int
    bytes_freed: int
```

### 3.4 Integration

Initial integration is **manual/explicit**, not automatic on every job:

1. Add a `scripts/cleanup_corpus_cache.py` entry point that:
   - Constructs `Settings()`.
   - Instantiates `CorpusCacheCleanupService`.
   - Runs `clean()` and logs a structured summary.
2. (Optional) Add a maintenance CLI command wired via `Makefile`, e.g. `make cleanup-corpus-cache`.

We deliberately avoid invoking cache cleanup from within training/eval jobs to keep job semantics simple and avoid hidden side effects. Cache cleanup is a **maintenance operation** with explicit telemetry.

### 3.5 Testing

Unit tests under `server/tests/core/services/data/test_corpus_cache_cleanup.py`:

- Empty cache dir: `clean()` returns zero deletions; no errors.
- Cache below thresholds: no deletion occurs.
- Cache above `max_bytes`: LRU eviction deletes enough files to drop below the limit.
- Free space below `min_free_bytes`: eviction continues until the guard is satisfied.
- Filesystem errors during `os.scandir` or `unlink` raise `CorpusCacheCleanupError`.
- Full coverage of statement and branch paths, including "no-op" and "error" branches.

Integration:

- A small test invoking the script/entry function with a temporary cache directory seeded with files, asserting:
  - Proper logging of deletions.
  - No side effects outside the cache directory.

---

## 4. Tokenizer Cleanup

### 4.1 Semantics

Tokenizer artifacts are **user-visible, versioned assets**:

- Local path: `artifacts_root/tokenizers/{tokenizer_id}`.
- They are consumed by:
  - Training jobs (`tokenizer_id` in `TrainRequest` and manifests).
  - Eval jobs (via manifests referencing `tokenizer_id`).
  - Potential inference services downstream.

Tokenizers should also be uploaded to data-bank-api and referenced by `file_id`, analogous to model artifacts. However, unlike corpus cache, tokenizers are not pure cache; they represent configuration and must only be deleted when provably unused in **this service**.

### 4.2 Configuration (`TokenizerCleanupConfig`)

Add a dedicated config:

```python
class TokenizerCleanupConfig(BaseSettings):
    enabled: bool = False
    min_unused_days: int = 30

    model_config = {
        "extra": "forbid",
        "env_nested_delimiter": "__",
    }
```

Integrated into `AppConfig` as `tokenizer_cleanup: TokenizerCleanupConfig`.

Semantics:

- Do not delete any tokenizer that:
  - Appears in any **existing manifest** under `artifacts_root/models/*/manifest.json`, or
  - Is referenced by any in-flight or queued job (checked via Redis and/or persistent queues).
- Only delete tokenizers that:
  - Are not referenced anywhere as above, and
  - Have not been modified or accessed in at least `min_unused_days` (based on `stat().st_mtime`).

### 4.3 Service (`TokenizerCleanupService`)

Location: `server/model_trainer/core/services/tokenizer/tokenizer_cleanup.py`

Responsibilities:

1. Discover existing tokenizer directories under `artifacts_root/tokenizers`.
2. Build the set of **tokenizer IDs in use** by:
   - Scanning manifests under `artifacts_root/models/*/manifest.json` and collecting `tokenizer_id` values.
   - Optionally scanning Redis for active job payloads (if they store `tokenizer_id`; otherwise, this can be omitted for the MVP and documented).
3. For each tokenizer directory:
   - If its `tokenizer_id` is in the "in use" set → skip.
   - Else, if its age is less than `min_unused_days` → skip.
   - Else, delete the tokenizer directory.

Error handling:

- Any IO/JSON parse errors during manifest scanning or deletion raise a typed `TokenizerCleanupError`.
- No partial/best-effort cleanup: either the cleanup run completes and returns a `TokenizerCleanupResult`, or it fails loudly and does **not** delete any further tokenizers after the first failure.

Result type:

```python
@dataclass
class TokenizerCleanupResult:
    deleted_tokenizers: int
    bytes_freed: int
```

### 4.4 Integration

As with corpus cache, integration is **explicit**:

- A maintenance script (`scripts/cleanup_tokenizers.py`) that:
  - Loads `Settings()`.
  - Instantiates `TokenizerCleanupService`.
  - Logs a summary of deleted tokenizers and freed bytes.

We do **not** delete tokenizers implicitly during training or eval to avoid surprising users; tokenizer cleanup is an operational decision with explicit tooling.

### 4.5 Testing

Unit tests under `server/tests/core/services/tokenizer/test_tokenizer_cleanup.py`:

- Tokenizer referenced in one or more manifests → never deleted.
- Tokenizer not referenced in any manifest, older than `min_unused_days` → deleted.
- Mixed case with several tokenizers; only unused/old ones are deleted.
- IO errors while reading manifests or deleting directories raise `TokenizerCleanupError`.

Integration:

- A small integration test that seeds a fake `artifacts_root/tokenizers` and `artifacts_root/models/*/manifest.json`, runs the cleanup script, and asserts the correct directories are removed.

---

## 5. Logging & Observability

For each cleanup operation (corpus cache and tokenizers):

- Emit structured logs with:
  - `event`: `"corpus_cache_cleanup_started"`, `"corpus_cache_cleanup_completed"`, `"corpus_cache_cleanup_failed"`, `"tokenizer_cleanup_started"`, etc.
  - `bytes_freed`, `deleted_files` / `deleted_tokenizers`.
  - Threshold configuration (`max_bytes`, `min_free_bytes`, `min_unused_days`) as context.
- On failure, include the error type and message; do not suppress exceptions.

These logs allow operators to:

- Track disk space recovery over time.
- Detect cleanup failures promptly.
- Tune thresholds based on observed behavior.

---

## 6. Rollout Strategy

1. **Phase 1**: Implement services and scripts, but keep both `corpus_cache_cleanup.enabled` and `tokenizer_cleanup.enabled` at their defaults (`False`).
   - Run cleanup scripts manually in staging to validate behavior and logging.
   - Confirm that no active tokenizers are incorrectly deleted and that cache cleanup obeys thresholds.

2. **Phase 2**: Enable corpus cache cleanup in production.
   - Set `APP__CORPUS_CACHE_CLEANUP__ENABLED=true` with conservative thresholds.
   - Run the maintenance script as part of periodic operational tasks.

3. **Phase 3**: Consider enabling tokenizer cleanup.
   - Only after verifying that manifests/tokenizer references are complete and correct.
   - Start with a large `min_unused_days` (e.g., 90) and monitor behavior.

---

## 7. Success Criteria

- Corpus cache size remains within configured bounds without impacting job correctness.
- Tokenizers that are truly unused (by Model-Trainer) and older than the configured window can be safely removed without breaking training or eval.
- No undiscovered "best-effort" branches: all cleanup failures are visible, logged, and raised as typed exceptions.
- 100% statements and branches coverage for `CorpusCacheCleanupService` and `TokenizerCleanupService` (and their helpers).
- Design docs (`CLEANUP_DESIGN.md`, this document) stay in sync with the code and tests as features evolve.

