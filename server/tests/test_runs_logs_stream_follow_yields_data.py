"""Test runs logs stream follow=True actually yields new data - covers runs.py:107."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import pytest
from model_trainer.api.routes import runs as runs_routes
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


class _FollowReader:
    """Mock file reader that simulates data arriving during follow mode."""

    def __init__(self: _FollowReader, new_data: list[bytes]) -> None:
        self._new_data: list[bytes] = new_data
        self._index: int = 0
        self._at_end: bool = False

    def __enter__(self: _FollowReader) -> _FollowReader:
        return self

    def __exit__(
        self: _FollowReader,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> Literal[False]:
        return False

    def seek(self: _FollowReader, offset: int, whence: int = 0) -> int:
        # Seek to end for follow mode - mark that we're at the end
        if whence == os.SEEK_END:
            self._at_end = True
        return 0

    def readline(self: _FollowReader) -> bytes:
        # Simulate follow mode: seek positions us at end, then data arrives
        # The follow loop in runs.py seeks to end, then repeatedly calls readline()
        # We want: first readline() -> new data (line 107 yields), next -> more data, etc.
        if self._at_end and self._index < len(self._new_data):
            line = self._new_data[self._index]
            self._index += 1
            return line
        # No more data
        return b""


def test_runs_logs_stream_follow_yields_new_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that follow=True actually yields data on line 107 of runs.py."""
    # Arrange artifacts and initial log file
    artifacts = tmp_path / "artifacts"
    os.environ["APP__ARTIFACTS_ROOT"] = str(artifacts)
    run_id = "run-follow-yield"
    run_dir = artifacts / "models" / run_id
    run_dir.mkdir(parents=True)
    log_file = run_dir / "logs.jsonl"
    log_file.write_text('{"initial":"line"}\n', encoding="utf-8")

    s = Settings()
    container = ServiceContainer.from_settings(s)
    h = runs_routes._RunsRoutes(container)

    # Configure controlled follow behavior
    h._sleep_fn = lambda _: None  # No-op sleep
    h._follow_max_loops = 3  # Limit loops to avoid infinite iteration

    call_count = {"n": 0}

    @contextmanager
    def _first_open_cm(p: str, m: str) -> Iterator[object]:
        """First open call for initial tail - use real file."""
        import io

        f = io.BytesIO(Path(p).read_bytes())
        try:
            yield f
        finally:
            f.close()

    def _open_monkey(path: str, mode: str = "r", *args: object, **kwargs: object) -> object:
        """Control open calls: first is real, second is mock with new data."""
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First open: initial tail read
            return _first_open_cm(path, mode)
        if path.endswith("logs.jsonl") and "rb" in mode:
            # Second open: follow mode with new data arriving
            return _FollowReader([b'{"new":"data1"}\n', b'{"new":"data2"}\n'])
        raise AssertionError(f"unexpected open call: {path}, {mode}")

    import model_trainer.api.routes.runs as runs_mod

    monkeypatch.setattr(runs_mod, "open", _open_monkey, raising=False)

    # Act: Drive the SSE iterator with follow=True
    gen = h._sse_iter(str(log_file), tail=1, follow=True)
    out: list[bytes] = list(gen)

    # Assert: should yield initial tail + new data from follow loop (line 107)
    assert len(out) >= 2  # At least initial + one follow yield
    # Check that we got SSE formatted data
    assert all(chunk.startswith(b"data: ") for chunk in out)
    # Verify we got the new data from the follow phase
    combined = b"".join(out)
    assert b"new" in combined or b"data1" in combined or b"data2" in combined
