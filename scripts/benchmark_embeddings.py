"""Benchmark different embedding models using vector retrieval only.

Purpose: isolate embedding quality without mixing lexical signals.
Outputs: results/benchmark_embeddings.json (+ optional CSV elsewhere).
"""
from __future__ import annotations

from pathlib import Path
import json
import sys
import time
import re
import unicodedata
from typing import Dict, Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import Embedder, EmbeddingCache
from src.eval_data import TEST_SET
from src.loaders.pdf_loader import load_pdf
from src.loaders.csv_loader import load_csv
from src.loaders.json_loader import load_json
from src.loaders.txt_loader import load_txt
from src.loaders.markdown_loader import load_markdown
from src.pipeline import clean_text, fixed_chunk, semantic_chunk, make_chunk_id
from src.rag.vectorstore import VectorStore
from src.rag.retriever import Retriever
from src.config import load_config

ROOT = Path("data")
RESULTS_DIR = Path("results")
INDEX_DIR = Path("index_benchmark")
K_VALUES = [4, 8, 12, 16]

LOADERS = {
    "pdf": load_pdf,
    "csv": load_csv,
    "json": load_json,
    "txt": load_txt,
    "md": load_markdown,
}

MODELS = [
    {
        "name": "ollama-nomic-embed-text",
        "backend": "ollama",
        "model_name": "nomic-embed-text:latest",
        "st_model": None,
    },
    {
        "name": "sentence-transformers-all-mpnet-base-v2",
        "backend": "sentence-transformers",
        "model_name": None,
        "st_model": "all-mpnet-base-v2",
    },
    {
        "name": "sentence-transformers-paraphrase-multilingual-mpnet-base-v2",
        "backend": "sentence-transformers",
        "model_name": None,
        "st_model": "paraphrase-multilingual-mpnet-base-v2",
    },
]


def iter_docs():
    """Yield normalized documents from all supported formats."""
    for ext, loader in LOADERS.items():
        for path in ROOT.rglob(f"*.{ext}"):
            for d in loader(path):
                doc_type = d.get("metadata", {}).get("type")
                preserve_newlines = doc_type in {"markdown", "csv", "json"}
                d["text"] = clean_text(d["text"], preserve_newlines=preserve_newlines)
                yield d


def chunk_doc(doc: dict) -> List[dict]:
    """Chunk by type to avoid mixing structured rows with narrative text."""
    doc_type = doc.get("metadata", {}).get("type")
    if doc_type in {"csv", "json"}:
        return fixed_chunk(doc, size=400, overlap=0)
    return semantic_chunk(doc, max_size=500, overlap=60)


def batched(iterable: Iterable[Tuple[str, str, dict]], batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_index(embedder: Embedder, index_dir: Path, cache_path: Path) -> VectorStore:
    """Build a temporary FAISS index for a given embedding model."""
    cache = EmbeddingCache(cache_path)
    store = None
    batch_size = 64
    total = 0

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
            # Only embed uncached chunks.
            new_embs = embedder.encode(missing_texts)
            cache.set_many({k: v.tolist() for k, v in zip(missing_ids, new_embs)})
            for pos, emb in zip(missing_positions, new_embs):
                embs[pos] = emb

        embs_arr = np.asarray(embs, dtype=np.float32)
        if store is None:
            store = VectorStore(dim=embs_arr.shape[1])
        store.add(ids=ids, embeddings=embs_arr, metadatas=mets, documents=texts)
        total += len(batch)
        cache.save()

    if store is None:
        raise RuntimeError("No se encontraron documentos en data/.")
    store.save(index_dir)
    return store


def normalize_text(text: str, strip_punct: bool = False) -> str:
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    if strip_punct:
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_retrieval(retriever: Retriever, k: int = 8, strip_punct: bool = False) -> Dict[str, float]:
    scores = []
    for test in TEST_SET:
        q = test["question"]
        expected = test["expected"]
        results = retriever.retrieve(q)
        results = results[:k]
        context = normalize_text(" ".join(r["text"] for r in results), strip_punct=strip_punct)
        expected_norm = [normalize_text(sub, strip_punct=strip_punct) for sub in expected]
        hits = sum(1 for sub in expected_norm if sub and sub in context)
        score = hits / len(expected) if expected else 0.0
        scores.append(score)
    avg = sum(scores) / len(scores) if scores else 0.0
    return {"avg_score": avg, "min_score": min(scores, default=0.0), "max_score": max(scores, default=0.0)}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"runs": [], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    cfg = load_config()
    strip_punct = bool(cfg.get("llm_similarity_strip_punct", False))

    for cfg in MODELS:
        name = cfg["name"]
        run_index_dir = INDEX_DIR / name
        run_cache = run_index_dir / "embeddings_cache.json"
        run_index_dir.mkdir(parents=True, exist_ok=True)

        embedder = Embedder(
            backend=cfg["backend"],
            model_name=cfg.get("model_name") or "nomic-embed-text:latest",
            st_model_name=cfg.get("st_model") or "all-mpnet-base-v2",
        )

        start = time.time()
        store = build_index(embedder, run_index_dir, run_cache)
        metrics_by_k = {}
        for k in K_VALUES:
            retriever = Retriever(embedder, store, k=k)
            metrics_by_k[str(k)] = score_retrieval(retriever, k=k, strip_punct=strip_punct)
        elapsed = time.time() - start

        summary["runs"].append(
            {
                "name": name,
                "backend": cfg["backend"],
                "model_name": cfg.get("model_name"),
                "st_model": cfg.get("st_model"),
                "metrics_by_k": metrics_by_k,
                "elapsed_seconds": round(elapsed, 2),
            }
        )
        best_k = max(metrics_by_k, key=lambda kk: metrics_by_k[kk]["avg_score"])
        best = metrics_by_k[best_k]
        print(
            f"{name}: best_k={best_k} avg={best['avg_score']:.3f} "
            f"min={best['min_score']:.3f} max={best['max_score']:.3f}"
        )

    out_path = RESULTS_DIR / "benchmark_embeddings.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Resumen guardado en {out_path.resolve()}")


if __name__ == "__main__":
    main()
