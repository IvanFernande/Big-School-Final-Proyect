import hashlib
from typing import Dict


def make_chunk_id(text: str, metadata: Dict) -> str:
    base = "|".join(
        [
            str(metadata.get("source", "")),
            str(metadata.get("type", "")),
            str(metadata.get("page", "")),
            str(metadata.get("row", "")),
            str(metadata.get("chunk_index", "")),
            str(metadata.get("start", "")),
            str(metadata.get("end", "")),
        ]
    )
    digest = hashlib.sha1((base + "|" + text).encode("utf-8")).hexdigest()[:16]
    return f"chunk-{digest}"
