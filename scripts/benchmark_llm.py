from __future__ import annotations

import json
import csv
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, get_secret
from src.embeddings import Embedder
from src.eval_data import TEST_SET
from src.rag.vectorstore import VectorStore
from src.rag.retriever import Retriever, HybridRetriever, KeywordRetriever, BM25Retriever, HybridBM25Retriever
from src.rag.reranker import CrossEncoderReranker

RESULTS_DIR = Path("results")


def build_prompt(question: str, contexts: List[Dict], max_contexts: int) -> str:
    selected = contexts[:max_contexts]
    context_text = "\n\n".join(f"- {c['text']}" for c in selected)
    return (
        "Responde de forma concisa basandote solo en el contexto. "
        "Incluye numeros tal cual aparecen. Si no esta en el contexto, di que no esta disponible.\n"
        f"Pregunta: {question}\n"
        f"Contexto:\n{context_text}\n"
        "Respuesta:"
    )


def ollama_generate(base_url: str, model: str, prompt: str, temperature: float) -> str:
    base = base_url.rstrip("/")
    resp = requests.post(
        f"{base}/api/generate",
        json={"model": model, "prompt": prompt, "temperature": temperature, "stream": False},
        timeout=120,
    )
    if resp.status_code == 404:
        # Fallback a /api/chat para versiones que no exponen /api/generate
        resp = requests.post(
            f"{base}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "stream": False,
            },
            timeout=120,
        )
    resp.raise_for_status()
    data = resp.json()
    if "response" in data:
        return (data.get("response") or "").strip()
    message = data.get("message", {})
    return (message.get("content") or "").strip()


def gemini_generate(model: str, prompt: str, temperature: float, api_key: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(model)
    response = llm.generate_content(prompt, generation_config={"temperature": temperature})
    return (response.text or "").strip()


def score_answer(answer: str, expected: List[str], strip_punct: bool) -> float:
    if not expected:
        return 0.0
    ans_norm = normalize_text(answer or "", strip_punct=strip_punct)
    expected_norm = [normalize_text(sub, strip_punct=strip_punct) for sub in expected]
    hits = sum(1 for sub in expected_norm if sub and sub in ans_norm)
    return hits / len(expected)


def context_coverage(contexts: List[Dict], expected: List[str], strip_punct: bool) -> float:
    if not expected:
        return 0.0
    text = normalize_text(" ".join(c["text"] for c in contexts), strip_punct=strip_punct)
    expected_norm = [normalize_text(sub, strip_punct=strip_punct) for sub in expected]
    hits = sum(1 for sub in expected_norm if sub and sub in text)
    return hits / len(expected)


def extract_entities(text: str) -> Dict[str, List[str]]:
    if not text:
        return {"emails": [], "incidents": [], "versions": []}
    emails = re.findall(r"[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}", text)
    incidents = re.findall(r"INC-\\d{4}-\\d{3}", text)
    versions = re.findall(r"\\b\\d+\\.\\d+\\.\\d+\\b", text)
    return {"emails": emails, "incidents": incidents, "versions": versions}


def groundedness(answer: str, contexts: List[Dict]) -> float:
    ctx_text = " ".join(c["text"] for c in contexts)
    entities = extract_entities(answer)
    total = sum(len(v) for v in entities.values())
    if total == 0:
        return 1.0
    missing = 0
    for group in entities.values():
        for ent in group:
            if ent not in ctx_text:
                missing += 1
    return 1.0 - (missing / total)


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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    embedder = Embedder(
        backend=cfg.get("embed_backend"),
        model_name=cfg.get("embed_model_name", "nomic-embed-text:latest"),
        st_model_name=cfg.get("st_model_name", "all-mpnet-base-v2"),
        device=cfg.get("embed_device"),
    )

    store = VectorStore.load(Path("index"))
    retriever_k = int(cfg.get("retriever_k", 12))
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

    models = cfg.get("llm_benchmark_models", [])
    temperature = float(cfg.get("llm_temperature", 0.2))
    max_contexts = int(cfg.get("llm_max_contexts", 8))
    limit = int(cfg.get("llm_benchmark_limit", 0))
    gemini_max_per_min = int(cfg.get("llm_gemini_max_per_min", 10))
    ollama_base_url = cfg.get("ollama_base_url", "http://localhost:11434")
    gemini_api_key = cfg.get("gemini_api_key") or get_secret("GEMINI_API_KEY") or ""

    summary = {"runs": [], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    detail_rows = []

    strip_punct = bool(cfg.get("llm_similarity_strip_punct", False))
    for model_cfg in models:
        name = model_cfg.get("name", "unnamed")
        provider = model_cfg.get("provider")
        model = model_cfg.get("model")
        if not provider or not model:
            print(f"[SKIP] Modelo mal configurado: {model_cfg}")
            continue

        scores = []
        coverages = []
        grounded = []
        similarities = []
        exact_matches = []
        retrieval_times = []
        generation_times = []
        total_times = []
        breakdown: Dict[str, Dict[str, float]] = {}
        breakdown_counts: Dict[str, int] = {}
        breakdown_sim_counts: Dict[str, int] = {}
        breakdown_exact_counts: Dict[str, int] = {}
        completeness = 0
        start = time.time()

        tests = TEST_SET[:limit] if limit and limit > 0 else TEST_SET
        wants_similarity = any(t.get("expected_text") or t.get("expect_text") for t in tests)
        similarity_model = None
        if wants_similarity:
            similarity_model = load_similarity_model(
                cfg.get("llm_similarity_model", cfg.get("st_model_name", "all-mpnet-base-v2")),
                cfg.get("embed_device"),
            )
        last_gemini_call = 0.0
        min_interval = 60.0 / gemini_max_per_min if gemini_max_per_min > 0 else 0.0

        for test in tests:
            q = test["question"]
            expected = test["expected"]
            expected_text = test.get("expected_text") or test.get("expect_text", "")
            category = test.get("category", "otros")
            t0 = time.perf_counter()
            contexts = retriever.retrieve(q)
            if reranker:
                contexts = reranker.rerank(q, contexts, top_k=rerank_top_k)
            t1 = time.perf_counter()
            prompt = build_prompt(q, contexts, max_contexts=max_contexts)

            if provider == "ollama":
                answer = ollama_generate(ollama_base_url, model, prompt, temperature)
            elif provider == "gemini":
                if not gemini_api_key:
                    print("[SKIP] Falta gemini_api_key.")
                    answer = ""
                else:
                    if min_interval > 0:
                        now = time.time()
                        wait = min_interval - (now - last_gemini_call)
                        if wait > 0:
                            time.sleep(wait)
                    answer = gemini_generate(model, prompt, temperature, gemini_api_key)
                    last_gemini_call = time.time()
            else:
                print(f"[SKIP] Provider no soportado: {provider}")
                answer = ""
            t2 = time.perf_counter()

            score = score_answer(answer, expected, strip_punct=strip_punct)
            coverage = context_coverage(contexts, expected, strip_punct=strip_punct)
            g_score = groundedness(answer, contexts)
            sim_score = None
            if similarity_model is not None and expected_text:
                norm_answer = normalize_text(answer, strip_punct=strip_punct)
                norm_expected = normalize_text(expected_text, strip_punct=strip_punct)
                sim_score = semantic_similarity(norm_answer, norm_expected, similarity_model)
                similarities.append(sim_score)
                exact_match = 1.0 if norm_answer == norm_expected else 0.0
                exact_matches.append(exact_match)
            scores.append(score)
            coverages.append(coverage)
            grounded.append(g_score)
            retrieval_times.append(t1 - t0)
            generation_times.append(t2 - t1)
            total_times.append(t2 - t0)
            if score == 1.0:
                completeness += 1
            breakdown_counts[category] = breakdown_counts.get(category, 0) + 1
            cat = breakdown.setdefault(
                category,
                {
                    "score_sum": 0.0,
                    "coverage_sum": 0.0,
                    "grounded_sum": 0.0,
                    "similarity_sum": 0.0,
                    "exact_match_sum": 0.0,
                    "complete_hits": 0,
                    "retrieval_s_sum": 0.0,
                    "generation_s_sum": 0.0,
                    "total_s_sum": 0.0,
                },
            )
            cat["score_sum"] += score
            cat["coverage_sum"] += coverage
            cat["grounded_sum"] += g_score
            if sim_score is not None:
                cat["similarity_sum"] += sim_score
                breakdown_sim_counts[category] = breakdown_sim_counts.get(category, 0) + 1
                cat["exact_match_sum"] += exact_match
                breakdown_exact_counts[category] = breakdown_exact_counts.get(category, 0) + 1
            if score == 1.0:
                cat["complete_hits"] += 1
            cat["retrieval_s_sum"] += (t1 - t0)
            cat["generation_s_sum"] += (t2 - t1)
            cat["total_s_sum"] += (t2 - t0)
            detail_rows.append(
                {
                    "model_name": name,
                    "provider": provider,
                    "model": model,
                    "question": q,
                    "expected": "|".join(expected),
                    "expected_text": expected_text,
                    "answer": answer,
                    "score": score,
                    "context_coverage": coverage,
                    "groundedness": g_score,
                    "semantic_similarity": "" if sim_score is None else round(sim_score, 4),
                    "exact_match_norm": "" if sim_score is None else int(exact_match),
                    "retrieval_s": round(t1 - t0, 4),
                    "generation_s": round(t2 - t1, 4),
                    "total_s": round(t2 - t0, 4),
                }
            )

        elapsed = time.time() - start
        total_tests = len(tests)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_coverage = sum(coverages) / len(coverages) if coverages else 0.0
        avg_grounded = sum(grounded) / len(grounded) if grounded else 0.0
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        avg_exact_match = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
        avg_retrieval_s = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0
        avg_generation_s = sum(generation_times) / len(generation_times) if generation_times else 0.0
        avg_total_s = sum(total_times) / len(total_times) if total_times else 0.0
        breakdown_out = {}
        for cat_name, stats in breakdown.items():
            count = breakdown_counts.get(cat_name, 1)
            breakdown_out[cat_name] = {
                "avg_score": stats["score_sum"] / count,
                "avg_context_coverage": stats["coverage_sum"] / count,
                "avg_groundedness": stats["grounded_sum"] / count,
                "avg_semantic_similarity": (
                    stats["similarity_sum"] / breakdown_sim_counts.get(cat_name, 1)
                    if breakdown_sim_counts.get(cat_name, 0) > 0
                    else 0.0
                ),
                "exact_match_rate": (
                    stats["exact_match_sum"] / breakdown_exact_counts.get(cat_name, 1)
                    if breakdown_exact_counts.get(cat_name, 0) > 0
                    else 0.0
                ),
                "completeness_rate": stats["complete_hits"] / count,
                "avg_retrieval_s": stats["retrieval_s_sum"] / count,
                "avg_generation_s": stats["generation_s_sum"] / count,
                "avg_total_s": stats["total_s_sum"] / count,
                "count": count,
            }
        summary["runs"].append(
            {
                "name": name,
                "provider": provider,
                "model": model,
                "avg_score": avg_score,
                "avg_context_coverage": avg_coverage,
                "avg_groundedness": avg_grounded,
                "avg_semantic_similarity": avg_similarity,
                "exact_match_rate": avg_exact_match,
                "completeness_rate": completeness / total_tests if total_tests else 0.0,
                "avg_retrieval_s": round(avg_retrieval_s, 4),
                "avg_generation_s": round(avg_generation_s, 4),
                "avg_total_s": round(avg_total_s, 4),
                "breakdown": breakdown_out,
                "elapsed_seconds": round(elapsed, 2),
            }
        )
        print(
            f"{name}: avg_score={avg_score:.3f} coverage={avg_coverage:.3f} grounded={avg_grounded:.3f} "
            f"semantic={avg_similarity:.3f} exact={avg_exact_match:.3f} complete={completeness/total_tests:.3f}"
        )

    out_path = RESULTS_DIR / "benchmark_llm.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Resumen guardado en {out_path.resolve()}")

    csv_path = RESULTS_DIR / "benchmark_llm.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "provider",
                "model",
                "question",
                "expected",
                "expected_text",
                "answer",
                "score",
                "context_coverage",
                "groundedness",
                "semantic_similarity",
                "exact_match_norm",
                "retrieval_s",
                "generation_s",
                "total_s",
            ],
        )
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"Detalle guardado en {csv_path.resolve()}")


if __name__ == "__main__":
    main()
