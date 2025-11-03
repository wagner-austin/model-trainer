from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass


@dataclass
class RunStore:
    artifacts_root: str

    def create_run(self: RunStore, model_family: str, model_size: str) -> str:
        ts = int(time.time())
        run_id = f"{model_family}-{model_size}-{ts}"
        # Create run directory under artifacts to avoid relying on a separate runs_root
        artifacts_dir = os.path.join(self.artifacts_root, "models", run_id)
        os.makedirs(artifacts_dir, exist_ok=True)
        # Write a small manifest for reproducibility and cross-links in the artifacts dir
        manifest_path = os.path.join(artifacts_dir, "manifest.json")
        body: dict[str, object] = {
            "run_id": run_id,
            "created_at": ts,
            "model_family": model_family,
            "model_size": model_size,
            "artifacts_dir": artifacts_dir,
            "logs_path": os.path.join(artifacts_dir, "logs.jsonl"),
            "status_key": f"runs:status:{run_id}",
            "heartbeat_key": f"runs:hb:{run_id}",
        }
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(body, f, separators=(",", ":"))
        except OSError as e:
            # Log and proceed; orchestrator/workers can still continue
            logging.getLogger(__name__).warning(
                "Failed to write run manifest path=%s error=%s", manifest_path, e
            )
        return run_id
