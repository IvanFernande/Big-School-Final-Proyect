from pathlib import Path
from typing import Dict, List
import PyPDF2

def load_pdf(path: Path) -> List[Dict]:
    docs = []
    with path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            docs.append({"text": text, "metadata": {"source": str(path), "page": i}})
    return docs
