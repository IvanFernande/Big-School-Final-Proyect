from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import Embedder
from src.rag.vectorstore import VectorStore
from src.rag.retriever import Retriever, HybridRetriever, KeywordRetriever
from src.rag.generator import SimpleGenerator
from src.eval_data import TEST_SET
from src.config import load_config

SLEEP_BETWEEN_REQUESTS = 5  # segundos entre preguntas para evitar 429
MAX_RETRIES = 3
RETRY_SLEEP = 60  # segundos para reintentos ante 429/quota


def score_answer(answer: str, expected_substrings) -> float:
    ans_low = (answer or "").lower()
    if not ans_low:
        return 0.0
    hits = sum(1 for sub in expected_substrings if sub.lower() in ans_low)
    return hits / len(expected_substrings) if expected_substrings else 0.0


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
    if retriever_type == "hybrid":
        alpha = float(cfg.get("retriever_alpha", 0.6))
        retriever = HybridRetriever(embedder, store, k=retriever_k, alpha=alpha)
    elif retriever_type == "keyword":
        retriever = KeywordRetriever(store, k=retriever_k)
    else:
        retriever = Retriever(embedder, store, k=retriever_k)
    try:
        generator = SimpleGenerator()
    except Exception as e:
        print(f"No se pudo inicializar el generador (Gemini). Configura GEMINI_API_KEY o config.json. Error: {e}")
        return

    results = []
    for i, test in enumerate(TEST_SET):
        q = test["question"]
        expected = test["expected"]
        retrieved = retriever.retrieve(q)
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
        else:
            score = score_answer(answer, expected)
        results.append((q, score, expected, answer))
        if i < len(TEST_SET) - 1:
            print(f"Esperando {SLEEP_BETWEEN_REQUESTS}s antes de la siguiente pregunta...")
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    print("Evaluacion RAG")
    print("-" * 80)
    for q, score, expected, answer in results:
        print(f"P: {q}")
        print(f"  Esperado: {expected}")
        print(f"  Respuesta: {answer}")
        print(f"  Score: {score:.2f}")
        print("-" * 80)


if __name__ == "__main__":
    main()
