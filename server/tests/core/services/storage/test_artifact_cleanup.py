from __future__ import annotations

from pathlib import Path
from typing import Final

import fakeredis
import pytest
from model_trainer.core.config.settings import CleanupConfig, Settings
from model_trainer.core.services.storage.artifact_cleanup import (
    ArtifactCleanupService,
    CleanupError,
)


def _settings_with_cleanup(
    enabled: bool,
    verify_upload: bool = True,
    dry_run: bool = False,
    grace_period_seconds: int = 0,
) -> Settings:
    settings = Settings()
    cleanup_cfg = CleanupConfig(
        enabled=enabled,
        verify_upload=verify_upload,
        grace_period_seconds=grace_period_seconds,
        dry_run=dry_run,
    )
    settings.app.cleanup = cleanup_cfg
    return settings


def _service(settings: Settings, redis_client: fakeredis.FakeRedis) -> ArtifactCleanupService:
    return ArtifactCleanupService(settings=settings, redis_client=redis_client)


def test_cleanup_disabled_returns_not_deleted(tmp_path: Path) -> None:
    settings = _settings_with_cleanup(enabled=False)
    r = fakeredis.FakeRedis(decode_responses=True)
    service = _service(settings, r)

    artifact_dir = tmp_path / "run"
    artifact_dir.mkdir()
    result = service.cleanup_run_artifacts("run-1", artifact_dir)

    assert result.deleted is False
    assert result.reason == "cleanup_disabled"
    assert artifact_dir.exists()


def test_cleanup_directory_not_found_returns_not_deleted(tmp_path: Path) -> None:
    settings = _settings_with_cleanup(enabled=True)
    r = fakeredis.FakeRedis(decode_responses=True)
    service = _service(settings, r)

    missing_dir = tmp_path / "missing"
    result = service.cleanup_run_artifacts("run-2", missing_dir)

    assert result.deleted is False
    assert result.reason == "directory_not_found"


def test_cleanup_no_file_id_in_redis_skips_deletion(tmp_path: Path) -> None:
    settings = _settings_with_cleanup(enabled=True)
    r = fakeredis.FakeRedis(decode_responses=True)
    # status is terminal but no file_id
    r.set("runs:status:run-3", "completed")
    artifact_dir = tmp_path / "run-3"
    artifact_dir.mkdir()

    service = _service(settings, r)
    result = service.cleanup_run_artifacts("run-3", artifact_dir)

    assert result.deleted is False
    assert result.reason == "upload_not_verified"
    assert artifact_dir.exists()


def test_cleanup_verify_upload_disabled_skips_redis_check(tmp_path: Path) -> None:
    settings = _settings_with_cleanup(enabled=True, verify_upload=False)
    r = fakeredis.FakeRedis(decode_responses=True)
    r.set("runs:status:run-4", "completed")
    artifact_dir = tmp_path / "run-4"
    artifact_dir.mkdir()

    service = _service(settings, r)
    result = service.cleanup_run_artifacts("run-4", artifact_dir)

    assert result.deleted is True
    assert result.reason is None
    assert not artifact_dir.exists()


def test_cleanup_skips_when_run_not_terminal(tmp_path: Path) -> None:
    settings = _settings_with_cleanup(enabled=True)
    r = fakeredis.FakeRedis(decode_responses=True)
    r.set("runs:artifact:run-5:file_id", "fid-123")
    r.set("runs:status:run-5", "running")
    artifact_dir = tmp_path / "run-5"
    artifact_dir.mkdir()

    service = _service(settings, r)
    result = service.cleanup_run_artifacts("run-5", artifact_dir)

    assert result.deleted is False
    assert result.reason == "run_not_terminal"
    assert artifact_dir.exists()


def test_cleanup_dry_run_does_not_delete(tmp_path: Path) -> None:
    settings = _settings_with_cleanup(enabled=True, dry_run=True)

    r = fakeredis.FakeRedis(decode_responses=True)
    r.set("runs:artifact:run-6:file_id", "fid-456")
    r.set("runs:status:run-6", "completed")
    artifact_dir = tmp_path / "run-6"
    artifact_dir.mkdir()
    (artifact_dir / "a.txt").write_text("x", encoding="utf-8")

    service = _service(settings, r)
    result = service.cleanup_run_artifacts("run-6", artifact_dir)

    assert result.deleted is False
    assert result.reason == "dry_run"
    assert artifact_dir.exists()


def test_cleanup_success_deletes_directory(tmp_path: Path) -> None:
    settings = _settings_with_cleanup(enabled=True)
    r = fakeredis.FakeRedis(decode_responses=True)
    r.set("runs:artifact:run-7:file_id", "fid-789")
    r.set("runs:status:run-7", "completed")
    artifact_dir = tmp_path / "run-7"
    artifact_dir.mkdir()
    f1 = artifact_dir / "a.txt"
    f2 = artifact_dir / "sub" / "b.txt"
    f2.parent.mkdir()
    f1.write_text("hello", encoding="utf-8")
    f2.write_text("world", encoding="utf-8")

    service = _service(settings, r)
    result = service.cleanup_run_artifacts("run-7", artifact_dir)

    assert result.deleted is True
    assert result.reason is None
    assert not artifact_dir.exists()
    assert result.files_deleted == 2
    assert result.bytes_freed >= 10


def test_cleanup_deletion_failure_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings_with_cleanup(enabled=True)
    r = fakeredis.FakeRedis(decode_responses=True)
    r.set("runs:artifact:run-8:file_id", "fid-000")
    r.set("runs:status:run-8", "completed")
    artifact_dir = tmp_path / "run-8"
    artifact_dir.mkdir()

    delete_called: Final[list[bool]] = [False]

    def _fail_rmtree(path: str) -> None:
        delete_called[0] = True
        raise OSError("boom")

    monkeypatch.setattr(
        "model_trainer.core.services.storage.artifact_cleanup.shutil.rmtree", _fail_rmtree
    )
    service = _service(settings, r)

    with pytest.raises(CleanupError):
        service.cleanup_run_artifacts("run-8", artifact_dir)

    assert delete_called[0] is True
    assert artifact_dir.exists()


def test_cleanup_grace_period_delays_before_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleep_called: Final[list[float]] = []

    def _fake_sleep(seconds: float) -> None:
        sleep_called.append(seconds)

    monkeypatch.setattr(
        "model_trainer.core.services.storage.artifact_cleanup.time.sleep", _fake_sleep
    )

    settings = _settings_with_cleanup(
        enabled=True,
        verify_upload=False,
        grace_period_seconds=1,
    )
    r = fakeredis.FakeRedis(decode_responses=True)
    r.set("runs:status:run-9", "completed")
    artifact_dir = tmp_path / "run-9"
    artifact_dir.mkdir()
    (artifact_dir / "a.txt").write_text("x", encoding="utf-8")

    service = _service(settings, r)
    result = service.cleanup_run_artifacts("run-9", artifact_dir)

    assert result.deleted is True
    assert not artifact_dir.exists()
    assert sleep_called == [1.0]


def test_calculate_size_and_count_handle_errors(tmp_path: Path) -> None:
    settings = _settings_with_cleanup(enabled=True, verify_upload=False)
    r = fakeredis.FakeRedis(decode_responses=True)
    r.set("runs:artifact:run-9:file_id", "fid-999")
    r.set("runs:status:run-9", "completed")
    artifact_dir = tmp_path / "run-9"
    artifact_dir.mkdir()
    (artifact_dir / "a.txt").write_text("x", encoding="utf-8")

    service = _service(settings, r)
    size = service._calculate_directory_size(artifact_dir)
    count = service._count_files(artifact_dir)

    assert size > 0
    assert count == 1
