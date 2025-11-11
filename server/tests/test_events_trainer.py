from __future__ import annotations

from model_trainer.events.trainer import CompletedV1, FailedV1, ProgressV1, StartedV1, encode_event


def test_encode_started() -> None:
    ev: StartedV1 = {
        "type": "trainer.train.started.v1",
        "request_id": "r1",
        "run_id": "run1",
        "user_id": 1,
        "model_family": "gpt2",
        "model_size": "small",
        "total_epochs": 1,
        "queue": "training",
    }
    s = encode_event(ev)
    assert "trainer.train.started.v1" in s and "\n" not in s


def test_encode_progress() -> None:
    ev: ProgressV1 = {
        "type": "trainer.train.progress.v1",
        "request_id": "r1",
        "run_id": "run1",
        "user_id": 1,
        "epoch": 1,
        "total_epochs": 1,
        "step": 10,
        "loss": 1.0,
    }
    s = encode_event(ev)
    assert "progress" in s and "loss" in s


def test_encode_completed_failed() -> None:
    done: CompletedV1 = {
        "type": "trainer.train.completed.v1",
        "request_id": "r1",
        "run_id": "run1",
        "user_id": 1,
        "loss": 0.5,
        "perplexity": 2.0,
        "artifact_path": "/x",
    }
    fail: FailedV1 = {
        "type": "trainer.train.failed.v1",
        "request_id": "r1",
        "run_id": "run1",
        "user_id": 1,
        "error_kind": "system",
        "message": "oom",
        "status": "failed",
    }
    assert "completed" in encode_event(done)
    assert "failed" in encode_event(fail)
