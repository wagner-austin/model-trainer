from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Final

RUNS_DIRNAME: Final[str] = "runs"


@dataclass
class RunStore:
    runs_root: str
    artifacts_root: str

    def create_run(self: RunStore, model_family: str, model_size: str) -> str:
        ts = int(time.time())
        run_id = f"{model_family}-{model_size}-{ts}"
        path = os.path.join(self.runs_root, run_id)
        os.makedirs(path, exist_ok=True)
        # Manifests can be added later; placeholder structure exists
        return run_id
