"""Run a small end-to-end evaluation over TEST_SET.

Uses current config for retrieval + generation and prints per-question scores.
"""
from pathlib import Path
import sys
import time
import unicodedata
import re

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import Embedder
from src.rag.vectorstore import VectorStore
from src.rag.retriever import Retriever, HybridRetriever, KeywordRetriever, BM25Retriever, HybridBM25Retriever
from src.rag.reranker import CrossEncoderReranker
from src.rag.generator import SimpleGenerator
from src.eval_data import TEST_SET
from src.config import load_config

SLEEP_BETWEEN_REQUESTS = 5  # segundos entre preguntas para evitar 429
MAX_RETRIES = 3
RETRY_SLEEP = 60  # segundos para reintentos ante 429/quota


def score_answer(answer: str, expected_substrings, strip_punct: bool) -> float:
    """Token-based expected match score."""
    ans_norm = normalize_text(answer or "", strip_punct=strip_punct)
    if not ans_norm:
        return 0.0
    expected_norm = [normalize_text(sub, strip_punct=strip_punct) for sub in expected_substrings]
    hits = sum(1 for sub in expected_norm if sub and sub in ans_norm)
    return hits / len(expected_substrings) if expected_substrings else 0.0


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


def load_similarity_model(model_name: str, device: str | None):
    """Lazy-load sentence-transformers model for semantic similarity."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        print(f"[WARN] sentence-transformers no disponible para similitud: {exc}")
        return None
    try:
        return SentenceTransformer(model_name, device=device or "cpu")
    except Exception as exc:
        print(f"[WARN] No se pudo cargar el modelo de similitud '{model_name}': {exc}")
        return None


def semantic_similarity(answer: str, expected_text: str, model) -> float:
    if not expected_text or not answer:
        return 0.0
    vectors = model.encode([answer, expected_text], normalize_embeddings=True)
    return float(np.dot(vectors[0], vectors[1]))


def main():
    index_dir = Path("index")
    try:
        store = VectorStore.load(index_dir)
    except FileNotFoundError:
        print(f"No se encontro un indice en {index_dir}. Ejecuta primero scripts/build_index.py.")
        return

    cfg = load_config()
    embedder = Embedder(
        backend=cfg.get("embed_backend"),
        model_name=cfg.get("embed_model_name", "nomic-embed-text:latest"),
        st_model_name=cfg.get("st_model_name", "all-mpnet-base-v2"),
        device=cfg.get("embed_device"),
    )
    retriever_k = int(cfg.get("retriever_k", 16))
    retriever_type = cfg.get("retriever_type", "vector")
    bm25_k1 = float(cfg.get("bm25_k1", 1.5))
    bm25_b = float(cfg.get("bm25_b", 0.75))
    if retriever_type == "hybrid":
        alpha = float(cfg.get("retriever_alpha", 0.6))
        retriever = HybridRetriever(embedder, store, k=retriever_k, alpha=alpha)
    elif retriever_type == "hybrid_bm25":
        alpha = float(cfg.get("retriever_alpha", 0.6))
        retriever = HybridBM25Retriever(embedder, store, k=retriever_k, alpha=alpha, k1=bm25_k1, b=bm25_b)
    elif retriever_type == "keyword":
        retriever = KeywordRetriever(store, k=retriever_k)
    elif retriever_type == "bm25":
        retriever = BM25Retriever(store, k=retriever_k, k1=bm25_k1, b=bm25_b)
    else:
        retriever = Retriever(embedder, store, k=retriever_k)
    reranker = None
    if cfg.get("rerank_enabled"):
        reranker = CrossEncoderReranker(
            model_name=cfg.get("rerank_model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            device=cfg.get("rerank_device") or cfg.get("embed_device"),
            batch_size=int(cfg.get("rerank_batch_size", 16)),
        )
    rerank_top_k = int(cfg.get("rerank_top_k", retriever_k))
    try:
        generator = SimpleGenerator()
    except Exception as e:
        print(f"No se pudo inicializar el generador (Gemini). Configura GEMINI_API_KEY o config.json. Error: {e}")
        return

    results = []
    strip_punct = bool(cfg.get("llm_similarity_strip_punct", False))
    wants_similarity = any(t.get("expected_text") or t.get("expect_text") for t in TEST_SET)
    similarity_model = None
    if wants_similarity:
        similarity_model = load_similarity_model(
            cfg.get("llm_similarity_model", cfg.get("st_model_name", "all-mpnet-base-v2")),
            cfg.get("embed_device"),
        )
    for i, test in enumerate(TEST_SET):
        q = test["question"]
        expected = test["expected"]
        expected_text = test.get("expected_text") or test.get("expect_text", "")
        retrieved = retriever.retrieve(q)
        if reranker:
            retrieved = reranker.rerank(q, retrieved, top_k=rerank_top_k)
        answer = None
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                answer = generator.answer(q, retrieved, max_contexts=8)
                break
            except Exception as e:
                last_error = e
                msg = str(e)
                if "ResourceExhausted" in msg or "429" in msg:
                    wait = RETRY_SLEEP * (attempt + 1)
                    print(f"[{q}] Cuota/429, reintentando en {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    break
        if answer is None:
            answer = f"[ERROR LLM] {last_error}"
            score = 0.0
            sim_score = None
            exact_match = None
        else:
            score = score_answer(answer, expected, strip_punct=strip_punct)
            sim_score = None
            if similarity_model is not None and expected_text:
                norm_answer = normalize_text(answer, strip_punct=strip_punct)
                norm_expected = normalize_text(expected_text, strip_punct=strip_punct)
                sim_score = semantic_similarity(norm_answer, norm_expected, similarity_model)
                exact_match = 1 if norm_answer == norm_expected else 0
        results.append((q, score, sim_score, exact_match, expected, expected_text, answer))
        if i < len(TEST_SET) - 1:
            print(f"Esperando {SLEEP_BETWEEN_REQUESTS}s antes de la siguiente pregunta...")
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    print("Evaluacion RAG")
    print("-" * 80)
    for q, score, sim_score, exact_match, expected, expected_text, answer in results:
        print(f"P: {q}")
        print(f"  Esperado: {expected}")
        if expected_text:
            print(f"  Esperado texto: {expected_text}")
        print(f"  Respuesta: {answer}")
        print(f"  Score: {score:.2f}")
        if sim_score is not None:
            print(f"  Similaridad: {sim_score:.3f}")
        if exact_match is not None:
            print(f"  Exact match: {exact_match}")
        print("-" * 80)


if __name__ == "__main__":
    main()
