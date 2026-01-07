from pathlib import Path
import sys
import time
import hashlib
from typing import Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders.pdf_loader import load_pdf
from src.loaders.csv_loader import load_csv
from src.loaders.json_loader import load_json
from src.loaders.txt_loader import load_txt
from src.loaders.markdown_loader import load_markdown
from src.pipeline import clean_text, fixed_chunk, semantic_chunk, make_chunk_id
from src.embeddings import Embedder, EmbeddingCache
from src.rag.vectorstore import VectorStore
from src.config import load_config

ROOT = Path("data")
INDEX_DIR = Path("index")
CACHE_PATH = INDEX_DIR / "embeddings_cache.json"
loaders = {
    "pdf": load_pdf,
    "csv": load_csv,
    "json": load_json,
    "txt": load_txt,
    "md": load_markdown
}


def iter_docs():
    for ext, loader in loaders.items():
        for path in ROOT.rglob(f"*.{ext}"):
            for d in loader(path):
                md = d.setdefault("metadata", {})
                md["filename"] = path.name
                md["ext"] = path.suffix.lstrip(".").lower()
                md["source_type"] = md.get("type", md["ext"])
                md["doc_id"] = f"{path.stem}:{hashlib.sha1(path.as_posix().encode('utf-8')).hexdigest()[:12]}"
                doc_type = md.get("type")
                preserve_newlines = doc_type in {"markdown", "txt", "pdf", "csv", "json"}
                d["text"] = clean_text(d["text"], preserve_newlines=preserve_newlines)
                yield d


def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def chunk_doc(doc: dict) -> List[dict]:
    doc_type = doc.get("metadata", {}).get("type")
    if doc_type in {"csv", "json"}:
        return fixed_chunk(doc, size=400, overlap=0)
    return semantic_chunk(doc, max_size=500, overlap=60)


def main():
    cfg = load_config()
    embedder = Embedder(
        backend=cfg.get("embed_backend"),
        model_name=cfg.get("embed_model_name", "nomic-embed-text:latest"),
        st_model_name=cfg.get("st_model_name", "all-mpnet-base-v2"),
        device=cfg.get("embed_device"),
    )
    cache = EmbeddingCache(CACHE_PATH)
    store = None
    batch_size = 64
    total = 0
    start_time = time.time()

    def chunk_iter() -> Iterable[Tuple[str, str, dict]]:
        for doc in iter_docs():
            for chunk in chunk_doc(doc):
                chunk_id = make_chunk_id(chunk["text"], chunk["metadata"])
                chunk["metadata"]["chunk_id"] = chunk_id
                yield chunk_id, chunk["text"], chunk["metadata"]

    for batch in batched(chunk_iter(), batch_size):
        ids = [b[0] for b in batch]
        texts = [b[1] for b in batch]
        mets = [b[2] for b in batch]

        embs: List[np.ndarray | list] = [None] * len(ids)
        missing_ids = []
        missing_texts = []
        missing_positions = []

        for i, chunk_id in enumerate(ids):
            cached = cache.get(chunk_id)
            if cached is not None:
                embs[i] = cached
            else:
                missing_ids.append(chunk_id)
                missing_texts.append(texts[i])
                missing_positions.append(i)

        if missing_texts:
            new_embs = embedder.encode(missing_texts)
            cache.set_many({k: v.tolist() for k, v in zip(missing_ids, new_embs)})
            for pos, emb in zip(missing_positions, new_embs):
                embs[pos] = emb

        if any(e is None for e in embs):
            raise RuntimeError("Embeddings incompletos en el batch.")

        embs_arr = np.asarray(embs, dtype=np.float32)
        if store is None:
            store = VectorStore(dim=embs_arr.shape[1])
        store.add(ids=ids, embeddings=embs_arr, metadatas=mets, documents=texts)
        total += len(batch)
        cache.save()
        print(f"Procesados {total} chunks (cache: {len(cache)}).")

    if store is None:
        print("No se encontraron documentos en data/.")
        return

    store.save(INDEX_DIR)
    elapsed = time.time() - start_time
    print(f"Indexados {total} chunks en {INDEX_DIR.resolve()} en {elapsed:.1f}s")


if __name__ == "__main__":
    main()
