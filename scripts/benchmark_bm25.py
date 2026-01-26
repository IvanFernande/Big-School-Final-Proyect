from __future__ import annotations

from pathlib import Path
import json
import sys
import time
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
from src.rag.retriever import BM25Retriever
from src.config import load_config

ROOT = Path("data")
RESULTS_DIR = Path("results")
INDEX_DIR = Path("index_benchmark")
K_VALUES = [4, 8, 12, 16]


def iter_docs():
    loaders = {
        "pdf": load_pdf,
        "csv": load_csv,
        "json": load_json,
        "txt": load_txt,
        "md": load_markdown,
    }
    for ext, loader in loaders.items():
        for path in ROOT.rglob(f"*.{ext}"):
            for d in loader(path):
                doc_type = d.get("metadata", {}).get("type")
                preserve_newlines = doc_type in {"markdown", "csv", "json"}
                d["text"] = clean_text(d["text"], preserve_newlines=preserve_newlines)
                yield d


def chunk_doc(doc: dict) -> List[dict]:
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
    cache = EmbeddingCache(cache_path)
    store = None
    batch_size = 64

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

        embs_arr = np.asarray(embs, dtype=np.float32)
        if store is None:
            store = VectorStore(dim=embs_arr.shape[1])
        store.add(ids=ids, embeddings=embs_arr, metadatas=mets, documents=texts)
        cache.save()

    if store is None:
        raise RuntimeError("No se encontraron documentos en data/.")
    store.save(index_dir)
    return store


def normalize_text(text: str, strip_punct: bool = False) -> str:
    if not text:
        return ""
    import re
    import unicodedata

    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    if strip_punct:
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_retrieval(retriever: BM25Retriever, k: int, strip_punct: bool) -> Dict[str, float]:
    scores = []
    by_cat: Dict[str, List[float]] = {}
    for test in TEST_SET:
        q = test["question"]
        expected = test["expected"]
        category = test.get("category", "otros")
        results = retriever.retrieve(q)[:k]
        context = normalize_text(" ".join(r["text"] for r in results), strip_punct=strip_punct)
        expected_norm = [normalize_text(sub, strip_punct=strip_punct) for sub in expected]
        hits = sum(1 for sub in expected_norm if sub and sub in context)
        score = hits / len(expected) if expected else 0.0
        scores.append(score)
        by_cat.setdefault(category, []).append(score)
    avg = sum(scores) / len(scores) if scores else 0.0
    breakdown = {}
    for cat, vals in by_cat.items():
        breakdown[cat] = {
            "avg_score": sum(vals) / len(vals) if vals else 0.0,
            "min_score": min(vals, default=0.0),
            "max_score": max(vals, default=0.0),
            "count": len(vals),
        }
    return {
        "avg_score": avg,
        "min_score": min(scores, default=0.0),
        "max_score": max(scores, default=0.0),
        "breakdown": breakdown,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    strip_punct = bool(cfg.get("llm_similarity_strip_punct", False))

    k1_grid = cfg.get("bm25_k1_grid") or [0.9, 1.2, 1.5, 1.8, 2.0]
    b_grid = cfg.get("bm25_b_grid") or [0.25, 0.5, 0.75, 0.9]

    embedder = Embedder(
        backend=cfg.get("embed_backend"),
        model_name=cfg.get("embed_model_name", "nomic-embed-text:latest"),
        st_model_name=cfg.get("st_model_name", "all-mpnet-base-v2"),
        device=cfg.get("embed_device"),
    )

    index_name = "bm25_benchmark"
    run_index_dir = INDEX_DIR / index_name
    run_cache = run_index_dir / "embeddings_cache.json"
    run_index_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    store = build_index(embedder, run_index_dir, run_cache)

    rows = []
    best = {"avg_score": -1, "k1": None, "b": None, "best_k": None}
    for k1 in k1_grid:
        for b in b_grid:
            metrics_by_k = {}
            for k in K_VALUES:
                retriever = BM25Retriever(store, k=k, k1=float(k1), b=float(b))
                metrics_by_k[str(k)] = score_retrieval(retriever, k=k, strip_punct=strip_punct)
            best_k = max(metrics_by_k, key=lambda kk: metrics_by_k[kk]["avg_score"])
            best_metrics = metrics_by_k[best_k]
            rows.append(
                {
                    "k1": float(k1),
                    "b": float(b),
                    "best_k": int(best_k),
                    "avg_score": best_metrics["avg_score"],
                    "min_score": best_metrics["min_score"],
                    "max_score": best_metrics["max_score"],
                    "breakdown": best_metrics.get("breakdown", {}),
                }
            )
            if best_metrics["avg_score"] > best["avg_score"]:
                best = {
                    "avg_score": best_metrics["avg_score"],
                    "k1": float(k1),
                    "b": float(b),
                    "best_k": int(best_k),
                }

    elapsed = time.time() - start
    summary = {
        "best": best,
        "grid": {"k1": k1_grid, "b": b_grid, "k_values": K_VALUES},
        "rows": rows,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = RESULTS_DIR / "benchmark_bm25.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"BM25 best: k1={best['k1']} b={best['b']} best_k={best['best_k']} avg={best['avg_score']:.3f}")
    print(f"Resumen guardado en {out_path.resolve()}")
    config_path = Path("config.json")
    if config_path.exists():
        try:
            cfg_data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            cfg_data = {}
        cfg_data["bm25_k1"] = best["k1"]
        cfg_data["bm25_b"] = best["b"]
        config_path.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")
        print(f"Config actualizado en {config_path.resolve()}")


if __name__ == "__main__":
    main()
