from typing import List, Dict

def fixed_chunk(doc: Dict, size: int = 400, overlap: int = 50) -> List[Dict]:
    text = doc["text"]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunk_text = text[start:end]
        chunks.append({"text": chunk_text, "metadata": {**doc["metadata"], "start": start, "end": end}})
        start = end - overlap
    return chunks
