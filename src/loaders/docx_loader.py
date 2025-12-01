from pathlib import Path
from typing import Dict, List
import docx

def load_docx(path: Path) -> List[Dict]:
    doc = docx.Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return [{"text": text, "metadata": {"source": str(path)}}]
