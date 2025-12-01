from pathlib import Path
from typing import Dict, List
import json

def load_json(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [{"text": json.dumps(data, ensure_ascii=False), "metadata": {"source": str(path)}}]
