from pathlib import Path
from typing import Dict, List, Any
import json


def _flatten(obj: Any, prefix: str = "") -> List[str]:
    lines: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}{k}"
            if isinstance(v, (dict, list)):
                lines.extend(_flatten(v, f"{new_prefix}."))
            else:
                lines.append(f"{new_prefix}: {v}")
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            new_prefix = f"{prefix}{idx}"
            if isinstance(v, (dict, list)):
                lines.extend(_flatten(v, f"{new_prefix}."))
            else:
                lines.append(f"{new_prefix}: {v}")
    else:
        lines.append(f"{prefix}: {obj}")
    return lines


def load_json(path: Path) -> List[Dict]:
    """Carga JSON y lo aplana por bloque logico (keys principales)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    docs: List[Dict] = []

    if isinstance(data, dict):
        for key, value in data.items():
            lines = _flatten({key: value})
            text = "\n".join(lines)
            docs.append(
                {
                    "text": text,
                    "metadata": {
                        "source": str(path),
                        "type": "json",
                        "section_title": str(key),
                        "headings_path": str(key),
                    },
                }
            )
    else:
        lines = _flatten(data)
        text = "\n".join(lines)
        docs.append({"text": text, "metadata": {"source": str(path), "type": "json"}})
    return docs
