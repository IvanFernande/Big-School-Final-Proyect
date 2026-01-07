from pathlib import Path
import sys

# Asegura que el proyecto esté en sys.path cuando se ejecuta como script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import Embedder
from src.rag.vectorstore import VectorStore
from src.rag.retriever import Retriever, HybridRetriever, KeywordRetriever
from src.rag.generator import SimpleGenerator
import requests
from src.config import load_config, get_secret


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
    cfg = load_config()
    embedder = Embedder(
        backend=cfg.get("embed_backend"),
        model_name=cfg.get("embed_model_name", "nomic-embed-text:latest"),
        st_model_name=cfg.get("st_model_name", "all-mpnet-base-v2"),
        device=cfg.get("embed_device"),
    )
    retriever_k = int(cfg.get("retriever_k", 8))
    retriever_type = cfg.get("retriever_type", "vector")
    if retriever_type == "hybrid":
        alpha = float(cfg.get("retriever_alpha", 0.6))
        retriever = HybridRetriever(embedder, store, k=retriever_k, alpha=alpha)
    elif retriever_type == "keyword":
        retriever = KeywordRetriever(store, k=retriever_k)
    else:
        retriever = Retriever(embedder, store, k=retriever_k)
    results = retriever.retrieve(question)
    results = boost_by_type(question, results)
    if not results:
        print("Sin resultados.")
        return

    print("Contextos recuperados:")
    for i, r in enumerate(results, 1):
        print(f"[{i}] score={r['score']:.3f} | {r['metadata']}\n{r['text'][:400]}...\n")

    def ollama_generate(base_url: str, model: str, prompt: str, temperature: float) -> str:
        base = base_url.rstrip("/")
        resp = requests.post(
            f"{base}/api/generate",
            json={"model": model, "prompt": prompt, "temperature": temperature, "stream": False},
            timeout=120,
        )
        if resp.status_code == 404:
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

    def build_prompt(q: str, ctxs, max_contexts: int) -> str:
        selected = ctxs[:max_contexts]
        context_text = "\n\n".join(f"- {c['text']}" for c in selected)
        return (
            "Responde de forma concisa basandote solo en el contexto. "
            "Incluye numeros tal cual aparecen. Si no esta en el contexto, di que no esta disponible.\n"
            f"Pregunta: {q}\n"
            f"Contexto:\n{context_text}\n"
            "Respuesta:"
        )

    try:
        llm_winner = cfg.get("llm_winner")
        temperature = float(cfg.get("llm_temperature", 0.2))
        max_contexts = int(cfg.get("llm_max_contexts", 8))
        prompt = build_prompt(question, results, max_contexts=max_contexts)

        if llm_winner:
            models = cfg.get("llm_benchmark_models", [])
            winner_cfg = next((m for m in models if m.get("name") == llm_winner), None)
            if winner_cfg and winner_cfg.get("provider") == "ollama":
                base_url = cfg.get("ollama_base_url", "http://localhost:11434")
                answer = ollama_generate(base_url, winner_cfg.get("model"), prompt, temperature)
            elif winner_cfg and winner_cfg.get("provider") == "gemini":
                api_key = cfg.get("gemini_api_key") or get_secret("GEMINI_API_KEY") or ""
                if not api_key:
                    raise ValueError("Falta GEMINI_API_KEY o config.json con gemini_api_key.")
                answer = gemini_generate(winner_cfg.get("model"), prompt, temperature, api_key)
            else:
                raise ValueError("llm_winner no encontrado en llm_benchmark_models.")
        else:
            gen = SimpleGenerator()
            answer = gen.answer(question, results, max_contexts=max_contexts)

        print("Respuesta generada:\n", answer)
    except Exception as e:
        print(f"No se pudo generar respuesta: {e}")


if __name__ == "__main__":
    main()
