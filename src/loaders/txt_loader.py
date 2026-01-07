from pathlib import Path
from typing import Dict, List


def load_txt(path: Path) -> List[Dict]:
    return [{"text": path.read_text(encoding="utf-8"), "metadata": {"source": str(path), "type": "txt"}}]
