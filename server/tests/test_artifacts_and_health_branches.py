from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.api.schemas.health import HealthzResponse
from model_trainer.core.config.settings import Settings


def test_healthz_logs_and_ready_branches(tmp_path: Path) -> None:
    os.environ["APP__ARTIFACTS_ROOT"] = str(tmp_path / "artifacts")
    app = create_app(Settings())
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    hz = HealthzResponse.model_validate_json(r.text)
    assert hz.status == "ok"


def test_artifacts_resolve_base_not_found(tmp_path: Path) -> None:
    os.environ["APP__ARTIFACTS_ROOT"] = str(tmp_path / "artifacts")
    app = create_app(Settings())
    client = TestClient(app)
    # Non-existent tokenizer id triggers 404 at resolve base check
    r = client.get("/artifacts/tokenizers/tid/download", params={"path": "tokenizer.json"})
    assert r.status_code == 404


def test_tokenizers_stats_branch(tmp_path: Path) -> None:
    import fakeredis
    from model_trainer.core.contracts.tokenizer import TokenizerTrainStats
    from model_trainer.core.services.container import ServiceContainer

    os.environ["APP__ARTIFACTS_ROOT"] = str(tmp_path / "artifacts")
    app = create_app(Settings())
    client = TestClient(app)
    cont: ServiceContainer = app.state.container
    fake = fakeredis.FakeRedis(decode_responses=True)
    cont.redis = fake
    tok_id = "t-123"
    stats = TokenizerTrainStats(coverage=1.0, oov_rate=0.0, token_count=1, char_coverage=1.0)
    fake.set(f"tokenizer:{tok_id}:status", "completed")
    fake.set(f"tokenizer:{tok_id}:stats", stats.model_dump_json())
    from model_trainer.api.schemas.tokenizers import TokenizerInfoResponse

    r = client.get(f"/tokenizers/{tok_id}")
    assert r.status_code == 200
    info = TokenizerInfoResponse.model_validate_json(r.text)
    assert info.status == "completed" and info.token_count == 1
