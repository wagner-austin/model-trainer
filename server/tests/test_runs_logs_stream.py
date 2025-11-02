from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.core.config.settings import Settings


def test_runs_logs_stream_initial_tail(tmp_path: Path) -> None:
    # Arrange artifacts and a run log with three lines
    artifacts = tmp_path / "artifacts"
    os.environ["APP__ARTIFACTS_ROOT"] = str(artifacts)
    run_id = "run-stream"
    log_dir = artifacts / "models" / run_id
    log_dir.mkdir(parents=True)
    lines = [b'{"msg":"one"}\n', b'{"msg":"two"}\n', b'{"msg":"three"}\n']
    (log_dir / "logs.jsonl").write_bytes(b"".join(lines))

    app = create_app(Settings())
    client = TestClient(app)

    # Act: stream with tail=2 and collect only the first two SSE lines
    collected: list[str] = []
    with client.stream(
        "GET",
        f"/runs/{run_id}/logs/stream",
        params={"tail": 2, "follow": False},
    ) as r:
        assert r.status_code == 200
        for raw in r.iter_lines():
            if not raw:
                continue
            text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="ignore")
            if text.startswith("data: "):
                collected.append(text[len("data: ") :])
                if len(collected) >= 2:
                    break

    # Assert: last two lines are emitted first in order
    assert len(collected) == 2
    assert '{"msg":"two"}' in collected[0]
    assert '{"msg":"three"}' in collected[1]
