from __future__ import annotations

import fakeredis
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.api.schemas.tokenizers import TokenizerInfoResponse
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


def test_tokenizers_get_status_and_stats() -> None:
    app = create_app(Settings())
    container: ServiceContainer = app.state.container
    fake = fakeredis.FakeRedis(decode_responses=True)
    container.redis = fake

    tok_id = "tok-xyz"
    fake.set(f"tokenizer:{tok_id}:status", "completed")
    fake.set(
        f"tokenizer:{tok_id}:stats",
        '{"coverage":0.9,"oov_rate":0.1,"token_count":1000,"char_coverage":0.8}',
    )

    client = TestClient(app)
    r = client.get(f"/tokenizers/{tok_id}")
    assert r.status_code == 200
    body = TokenizerInfoResponse.model_validate_json(r.text)
    assert body.tokenizer_id == tok_id
    assert body.status == "completed"
    assert body.coverage == 0.9
    assert body.oov_rate == 0.1
    assert body.token_count == 1000
    assert body.char_coverage == 0.8
