from __future__ import annotations

import os
from collections import deque
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse

from ...core.infra.paths import model_logs_path
from ...core.logging.types import LoggingExtra
from ...core.services.container import ServiceContainer
from ..schemas.runs import (
    CancelResponse,
    EvaluateRequest,
    EvaluateResponse,
    RunStatusResponse,
    TrainRequest,
    TrainResponse,
)


class _RunsRoutes:
    c: ServiceContainer

    def __init__(self: _RunsRoutes, container: ServiceContainer) -> None:
        self.c = container

    def start_training(self: _RunsRoutes, req: TrainRequest) -> TrainResponse:
        orchestrator = self.c.training_orchestrator
        extra: LoggingExtra = {
            "event": "runs_enqueue",
            "model_family": req.model_family,
            "model_size": req.model_size,
        }
        self.c.logging.adapter(category="api", service="runs", run_id=None).info(
            "runs enqueue", extra=extra
        )
        out = orchestrator.enqueue_training(req)
        return TrainResponse(run_id=out.run_id, job_id=out.job_id)

    def run_status(self: _RunsRoutes, run_id: str) -> RunStatusResponse:
        orchestrator = self.c.training_orchestrator
        return orchestrator.get_status(run_id)

    def run_evaluate(self: _RunsRoutes, run_id: str, req: EvaluateRequest) -> EvaluateResponse:
        orchestrator = self.c.training_orchestrator
        extra2: LoggingExtra = {"event": "runs_enqueue_eval", "split": req.split}
        self.c.logging.adapter(category="api", service="runs", run_id=run_id).info(
            "runs enqueue eval", extra=extra2
        )
        return orchestrator.enqueue_evaluation(run_id, req)

    def run_eval_result(self: _RunsRoutes, run_id: str) -> EvaluateResponse:
        orchestrator = self.c.training_orchestrator
        result: EvaluateResponse = orchestrator.get_evaluation(run_id)
        return result

    def run_logs(self: _RunsRoutes, run_id: str, tail: int = 200) -> PlainTextResponse:
        path = str(model_logs_path(self.c.settings, run_id))
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="logs not found")
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            tail_n = max(1, int(tail))
            content = "".join(lines[-tail_n:])
            extra3: LoggingExtra = {"event": "runs_logs", "tail": tail_n}
            self.c.logging.adapter(category="api", service="runs", run_id=run_id).info(
                "runs logs", extra=extra3
            )
            return PlainTextResponse(content)
        except OSError:
            raise HTTPException(status_code=500, detail="failed to read logs") from None

    def run_logs_stream(
        self: _RunsRoutes,
        run_id: str,
        tail: int = 200,
        follow: bool = Query(True, description="Follow the log file for new lines"),
    ) -> StreamingResponse:
        path = str(model_logs_path(self.c.settings, run_id))
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="logs not found")

        sse_logger = self.c.logging.adapter(category="api", service="runs", run_id=run_id)

        def _iter_sse_lines() -> Iterator[bytes]:
            try:
                # Emit last `tail` lines immediately
                with open(path, "rb") as f:
                    last: deque[bytes] = deque(maxlen=max(1, int(tail)))
                    for line in f:
                        last.append(line)
                for line in last:
                    yield b"data: " + line.rstrip(b"\n") + b"\n\n"
                if not follow:
                    return
                # Follow the file
                with open(path, "rb") as f2:
                    f2.seek(0, os.SEEK_END)
                    while True:
                        chunk = f2.readline()
                        if chunk:
                            yield b"data: " + chunk.rstrip(b"\n") + b"\n\n"
                        else:
                            # Sleep a bit to avoid busy wait
                            import time as _time

                            _time.sleep(0.5)
            except OSError as e:
                # Log and stop streaming on file errors
                sse_logger.error(
                    "SSE file read error",
                    extra={
                        "event": "runs_logs_stream_error",
                        "reason": str(e),
                    },
                )
                return

        headers = {"Cache-Control": "no-cache"}
        extra4: LoggingExtra = {"event": "runs_logs_stream", "tail": max(1, int(tail))}
        sse_logger.info("runs logs stream", extra=extra4)
        return StreamingResponse(_iter_sse_lines(), media_type="text/event-stream", headers=headers)

    def cancel_run(self: _RunsRoutes, run_id: str) -> CancelResponse:
        r = self.c.redis
        r.set(f"runs:{run_id}:cancelled", "1")
        extra4: LoggingExtra = {"event": "runs_cancel"}
        self.c.logging.adapter(category="api", service="runs", run_id=run_id).info(
            "runs cancel", extra=extra4
        )
        return CancelResponse(status="cancellation-requested")


def build_router(container: ServiceContainer) -> APIRouter:
    router = APIRouter()
    h = _RunsRoutes(container)
    router.add_api_route(
        "/train",
        h.start_training,
        methods=["POST"],
        response_model=TrainResponse,
    )
    router.add_api_route(
        "/{run_id}",
        h.run_status,
        methods=["GET"],
        response_model=RunStatusResponse,
    )
    router.add_api_route(
        "/{run_id}/evaluate",
        h.run_evaluate,
        methods=["POST"],
        response_model=EvaluateResponse,
    )
    router.add_api_route(
        "/{run_id}/eval",
        h.run_eval_result,
        methods=["GET"],
        response_model=EvaluateResponse,
    )
    router.add_api_route(
        "/{run_id}/logs",
        h.run_logs,
        methods=["GET"],
    )
    router.add_api_route(
        "/{run_id}/logs/stream",
        h.run_logs_stream,
        methods=["GET"],
    )
    router.add_api_route(
        "/{run_id}/cancel",
        h.cancel_run,
        methods=["POST"],
        response_model=CancelResponse,
    )
    return router
