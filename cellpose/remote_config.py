import json
import os
from pathlib import Path
from typing import Any, Dict


def load_remote_config() -> Dict[str, Any]:
    env_path = os.environ.get("CELLPOSE_REMOTE_CONFIG")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "remote_config.json")
    candidates.append(Path.home() / ".cellpose" / "remote_config.json")
    for path in candidates:
        try:
            if path and path.exists():
                with open(path, "r", encoding="utf-8") as fh:
                    return json.load(fh) or {}
        except Exception:
            continue
    return {}
