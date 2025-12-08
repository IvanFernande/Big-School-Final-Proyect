from pathlib import Path
import sys

# Asegura que el proyecto esté en sys.path cuando se ejecuta como script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import Embedder
from src.vectorstore import VectorStore
from src.retriever import Retriever
from src.generator import SimpleGenerator


def boost_by_type(question: str, results):
    """Heurística sencilla: prioriza CSV para KPIs y JSON para configuración."""
    q_lower = question.lower()
    prefer_csv = any(k in q_lower for k in ["q1", "q2", "q3", "q4", "trimestre", "sla", "csat", "tickets"])
    prefer_json = any(k in q_lower for k in ["config", "parámetro", "parametro", "json", "objetivo"])
    def score(item):
        base = item["score"]
        meta = item.get("metadata", {})
        t = meta.get("type")
        if prefer_csv and t == "csv":
            base += 0.1
        if prefer_json and t == "json":
            base += 0.1
        return base
    return sorted(results, key=score, reverse=True)


def main():
    question = input("Pregunta: ")
    index_dir = Path("index")
    try:
        store = VectorStore.load(index_dir)
    except FileNotFoundError:
        print(f"No se encontró un índice en {index_dir}. Ejecuta primero scripts/build_index.py.")
        return
    embedder = Embedder()
    retriever = Retriever(embedder, store, k=8)
    results = retriever.retrieve(question)
    results = boost_by_type(question, results)
    if not results:
        print("Sin resultados.")
        return

    print("Contextos recuperados:")
    for i, r in enumerate(results, 1):
        print(f"[{i}] score={r['score']:.3f} | {r['metadata']}\n{r['text'][:400]}...\n")

    try:
        gen = SimpleGenerator()
        answer = gen.answer(question, results, max_contexts=8)
        print("Respuesta generada:\n", answer)
    except Exception as e:
        print(f"No se pudo generar respuesta (configura GEMINI_API_KEY o config.json): {e}")


if __name__ == "__main__":
    main()
