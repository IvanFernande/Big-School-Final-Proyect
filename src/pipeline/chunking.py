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
        chunks.append(
            {
                "text": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "start": start,
                    "end": end,
                    "char_start": start,
                    "char_end": end,
                },
            }
        )
        if end == len(text):
            break
        start += step
    return chunks


def semantic_chunk(
    doc: Dict,
    max_size: int = 500,
    overlap: int = 60,
    max_chars: int = 200_000,
) -> List[Dict]:
    """Chunking por secciones (headings) + ventana con solape."""
    text = doc["text"][:max_chars]
    lines = text.splitlines()
    sections: List[Dict] = []
    headings_stack: List[str] = []
    buffer: List[str] = []

    def flush_section():
        if buffer:
            body = " ".join(buffer).strip()
            section_title = headings_stack[-1] if headings_stack else ""
            headings_path = " > ".join(headings_stack) if headings_stack else ""
            sections.append(
                {
                    "text": body,
                    "section_title": section_title,
                    "headings_path": headings_path,
                }
            )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                buffer.append("")
            continue
        if stripped.startswith("#"):
            flush_section()
            buffer = []
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip()
            if level <= 0:
                continue
            if level <= len(headings_stack):
                headings_stack[:] = headings_stack[: level - 1]
            headings_stack.append(title)
            continue
        if stripped.startswith(("- ", "* ")):
            buffer.append(stripped[2:].strip())
            continue
        buffer.append(stripped)

    flush_section()

    chunks: List[Dict] = []
    chunk_index = 0
    for section in sections:
        section_text = section["text"]
        section_title = section["section_title"]
        headings_path = section["headings_path"]
        if len(section_text) <= max_size:
            chunks.append(
                {
                    "text": section_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": chunk_index,
                        "char_len": len(section_text),
                        "char_start": 0,
                        "char_end": len(section_text),
                        "section_title": section_title,
                        "headings_path": headings_path,
                    },
                }
            )
            chunk_index += 1
            continue
        start = 0
        step = max(max_size - overlap, 1)
        while start < len(section_text):
            end = min(len(section_text), start + max_size)
            chunk_text = section_text[start:end]
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": chunk_index,
                        "char_len": len(chunk_text),
                        "char_start": start,
                        "char_end": end,
                        "section_title": section_title,
                        "headings_path": headings_path,
                    },
                }
            )
            chunk_index += 1
            if end == len(section_text):
                break
            start += step

    return chunks
