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
from src.rag.retriever import Retriever, HybridRetriever, KeywordRetriever
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


def _chunk_relevance(text: str, expected: List[str]) -> float:
    if not expected:
        return 0.0
    text_low = text.lower()
    hits = sum(1 for sub in expected if sub.lower() in text_low)
    return hits / len(expected)


def score_retrieval(retriever, k: int = 8) -> Dict[str, float]:
    avg_scores = []
    recall_hits = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0

    for test in TEST_SET:
        q = test["question"]
        expected = test["expected"]
        results = retriever.retrieve(q)[:k]

        # average score (matching substrings in concatenated context)
        context = " ".join(r["text"] for r in results).lower()
        hits = sum(1 for sub in expected if sub.lower() in context)
        avg_score = hits / len(expected) if expected else 0.0
        avg_scores.append(avg_score)

        # relevance per chunk (graded)
        rels = [_chunk_relevance(r["text"], expected) for r in results]

        # Recall@K: any relevant chunk
        if any(r > 0 for r in rels):
            recall_hits += 1

        # MRR@K: rank of first relevant chunk
        first_rel_rank = next((i + 1 for i, r in enumerate(rels) if r > 0), None)
        if first_rel_rank is not None:
            mrr_sum += 1.0 / first_rel_rank

        # nDCG@K (graded)
        dcg = 0.0
        for i, rel in enumerate(rels):
            dcg += rel / np.log2(i + 2)
        ideal_rels = sorted(rels, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_rels):
            idcg += rel / np.log2(i + 2)
        if idcg > 0:
            ndcg_sum += dcg / idcg


    n = len(TEST_SET)
    avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0
    recall_at_k = recall_hits / n if n else 0.0
    mrr_at_k = mrr_sum / n if n else 0.0
    ndcg_at_k = ndcg_sum / n if n else 0.0

    return {
        "avg_score": avg,
        "min_score": min(avg_scores, default=0.0),
        "max_score": max(avg_scores, default=0.0),
        "recall_at_k": recall_at_k,
        "mrr_at_k": mrr_at_k,
        "ndcg_at_k": ndcg_at_k,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    embedder = Embedder(
        backend=cfg.get("embed_backend"),
        model_name=cfg.get("embed_model_name", "nomic-embed-text:latest"),
        st_model_name=cfg.get("st_model_name", "all-mpnet-base-v2"),
        device=cfg.get("embed_device"),
    )

    index_name = "retrieval_benchmark"
    run_index_dir = INDEX_DIR / index_name
    run_cache = run_index_dir / "embeddings_cache.json"
    run_index_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    store = build_index(embedder, run_index_dir, run_cache)

    modes = {
        "vector": lambda k: Retriever(embedder, store, k=k),
        "keyword": lambda k: KeywordRetriever(store, k=k),
        "hybrid": lambda k: HybridRetriever(embedder, store, k=k, alpha=0.6),
    }

    summary = {
        "runs": [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for mode_name, factory in modes.items():
        metrics_by_k = {}
        for k in K_VALUES:
            retriever = factory(k)
            metrics_by_k[str(k)] = score_retrieval(retriever, k=k)
        summary["runs"].append({"mode": mode_name, "metrics_by_k": metrics_by_k})
        best_k = max(metrics_by_k, key=lambda kk: metrics_by_k[kk]["avg_score"])
        best = metrics_by_k[best_k]
        print(
            f"{mode_name}: best_k={best_k} avg={best['avg_score']:.3f} "
            f"min={best['min_score']:.3f} max={best['max_score']:.3f}"
        )

    elapsed = time.time() - start
    summary["elapsed_seconds"] = round(elapsed, 2)

    out_path = RESULTS_DIR / "benchmark_retrieval.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Resumen guardado en {out_path.resolve()}")


if __name__ == "__main__":
    main()
