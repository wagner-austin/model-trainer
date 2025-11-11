from __future__ import annotations

import fakeredis
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.api.schemas.tokenizers import TokenizerInfoResponse
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


def test_tokenizers_get_unknown_status_and_no_stats() -> None:
    app = create_app(Settings())
    container: ServiceContainer = app.state.container
    # Use isolated fake redis with no keys set
    container.redis = fakeredis.FakeRedis(decode_responses=True)
    client = TestClient(app)

    tok_id = "tok-unknown"
    r = client.get(f"/tokenizers/{tok_id}")
    assert r.status_code == 200
    body = TokenizerInfoResponse.model_validate_json(r.text)
    assert body.tokenizer_id == tok_id
    assert body.status == "unknown"
    assert body.coverage is None and body.oov_rate is None and body.token_count is None
