from pathlib import Path
from typing import Dict, List


def load_markdown(path: Path) -> List[Dict]:
    """Load a Markdown/README file preserving headings for better downstream analysis."""
    text = path.read_text(encoding="utf-8")
    return [{"text": text, "metadata": {"source": str(path), "type": "markdown"}}]
