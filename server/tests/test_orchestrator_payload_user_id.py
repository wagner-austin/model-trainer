from __future__ import annotations

import fakeredis
from _pytest.monkeypatch import MonkeyPatch
from model_trainer.api.schemas.runs import TrainRequest
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from model_trainer.orchestrators.training_orchestrator import TrainingOrchestrator


def test_orchestrator_threads_user_id(monkeypatch: MonkeyPatch) -> None:
    s = Settings()
    r = fakeredis.FakeRedis(decode_responses=True)
    # Real enqueuer instance for type compatibility
    enq = RQEnqueuer(
        redis_url="redis://localhost:6379/0",
        settings=RQSettings(
            queue_name="training",
            job_timeout_sec=60,
            result_ttl_sec=60,
            failure_ttl_sec=60,
            retry_max=0,
            retry_intervals=[],
        ),
    )
    seen: dict[str, int] = {}

    def _fake_enqueue_train(payload: dict[str, object]) -> str:
        val = payload.get("user_id")
        assert isinstance(val, int)
        seen["user_id"] = val
        return "job-1"

    monkeypatch.setattr(enq, "enqueue_train", _fake_enqueue_train)
    orch = TrainingOrchestrator(settings=s, redis_client=r, enqueuer=enq, model_registry=None)
    # Stub CorpusFetcher to avoid network and provide a local path
    import tempfile
    from pathlib import Path

    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: object, *args: object, **kwargs: object) -> None:
            pass

        def fetch(self: object, file_id: str) -> Path:
            return Path(tempfile.gettempdir())

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)
    req = TrainRequest(
        model_family="gpt2",
        model_size="small",
        max_seq_len=128,
        num_epochs=1,
        batch_size=2,
        learning_rate=5e-4,
        corpus_file_id="deadbeef",
        tokenizer_id="tok1",
        user_id=42,
    )
    out = orch.enqueue_training(req)
    assert out.run_id and seen.get("user_id") == 42
