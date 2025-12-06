from typing import Dict, List


def fixed_chunk(doc: Dict, size: int = 400, overlap: int = 50, max_chars: int = 200_000) -> List[Dict]:
    """Divide el texto en bloques solapados; evita bucles infinitos en textos cortos."""
    text = doc["text"][:max_chars]
    chunks: List[Dict] = []
    start = 0
    step = max(size - overlap, 1)
    while start < len(text):
        end = min(len(text), start + size)
        chunk_text = text[start:end]
        chunks.append({"text": chunk_text, "metadata": {**doc["metadata"], "start": start, "end": end}})
        if end == len(text):
            break
        start += step
    return chunks
