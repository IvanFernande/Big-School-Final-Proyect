import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG = Path("config.json")


def load_config(path: str | Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_secret(key: str, path: str | Path = DEFAULT_CONFIG) -> Optional[str]:
    return os.getenv(key) or load_config(path).get(key)


def get_config(key: str, default: Any = None, path: str | Path = DEFAULT_CONFIG) -> Any:
    return load_config(path).get(key, default)
