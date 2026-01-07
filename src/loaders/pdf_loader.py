from pathlib import Path
from typing import Dict, List
import pdfplumber


def load_pdf(path: Path) -> List[Dict]:
    """Extract text and tables from each page of a PDF."""
    docs: List[Dict] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            table_sections = []
            for t in tables:
                rows = [",".join(cell or "" for cell in row) for row in t]
                table_sections.append("\n".join(rows))
            combined = text
            if table_sections:
                combined += "\n\n[TABLAS]\n" + "\n\n".join(table_sections)
            docs.append({"text": combined, "metadata": {"source": str(path), "page": i, "type": "pdf"}})
    return docs
