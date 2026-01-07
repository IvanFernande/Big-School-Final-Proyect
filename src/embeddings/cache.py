from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


class EmbeddingCache:
    """Cache simple en disco para embeddings por id estable."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: Dict[str, List[float]] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def get(self, key: str) -> Optional[List[float]]:
        return self._data.get(key)

    def set_many(self, items: Dict[str, List[float]]) -> None:
        self._data.update(items)

    def save(self) -> None:
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._data)
