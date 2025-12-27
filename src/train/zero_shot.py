"""
Demostración zero-shot con N_EXAMPLES ejemplos (Body, Department, Priority) y 1 ticket a predecir.
Usa Gemini 2.5 Flash (google-genai). Solo hace 1 inferencia.

Instrucciones:
- Configura GEMINI_API_KEY en .env o en DEFAULT_API_KEY.
- Ejecuta: python -m src.train.zero_shot
"""

import os
from pathlib import Path

import google.genai as genai
import requests
import numpy as np
from dotenv import load_dotenv

from src.data_prep import add_features, load, split

# Configuración por defecto
DEFAULT_API_KEY = "PON_AQUI_TU_API_KEY"
GEMINI_MODEL = "gemini-2.5-flash"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
PROVIDER = "ollama"
N_EXAMPLES = 5  # ejemplos en el prompt
RANDOM_SEED = 42


def build_fewshot_prompt(examples, query_body, query_dept):
    lines = [
        "Eres un asistente que clasifica tickets en: high, medium o low.",
        "Ejemplos:",
    ]
    for i, (body, dept, prio) in enumerate(examples, 1):
        lines.append(f"Ejemplo {i}:")
        lines.append(f"Department: {dept}")
        lines.append(f"Body: {body}")
        lines.append(f"Priority: {prio}")
        lines.append("")

    lines.append("Ahora clasifica este ticket:")
    lines.append(f"Department: {query_dept}")
    lines.append(f"Body: {query_body}")
    lines.append("Priority (responde solo high, medium o low):")
    return "\n".join(lines)


def main():
    # Carga .env
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=repo_root / ".env")
    provider = PROVIDER
    gemini_api_key = os.getenv("GEMINI_API_KEY", DEFAULT_API_KEY)
    if provider == "gemini":
        if not gemini_api_key or "PON_AQUI" in gemini_api_key:
            raise RuntimeError("Configura GEMINI_API_KEY en .env o en DEFAULT_API_KEY.")
        client = genai.Client(api_key=gemini_api_key)
    else:
        client = None

    # Prepara datos: toma N_EXAMPLES y 1 muestra a predecir (la siguiente)
    df = add_features(load())
    _, X_test, _, y_test = split(df)

    # Selección aleatoria reproducible
    rng = np.random.default_rng(RANDOM_SEED)
    sample_indices = rng.choice(len(X_test), size=N_EXAMPLES + 1, replace=False)
    example_indices = sample_indices[:-1]
    query_index = sample_indices[-1]

    examples = list(
        zip(
            X_test.iloc[example_indices]["Body"].astype(str),
            X_test.iloc[example_indices]["Department"].astype(str),
            y_test.iloc[example_indices].astype(str),
        )
    )
    query_row = X_test.iloc[query_index]
    query_body = str(query_row["Body"])
    query_dept = str(query_row["Department"])
    query_true = str(y_test.iloc[query_index])

    prompt = build_fewshot_prompt(examples, query_body, query_dept)
    print("====== PROMPT ENVIADO ======")
    print(prompt)
    print("============================")
    print(f"Prioridad esperada (no enviada al LLM): {query_true}")

    if provider == "gemini":
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        answer_raw = resp.text or ""
    elif provider == "ollama":
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        answer_raw = resp.json().get("response", "")
    else:
        raise RuntimeError(f"Proveedor no soportado: {provider}")

    answer = answer_raw.strip().lower()
    print(f"Proveedor: {provider}")
    print("Respuesta cruda del modelo:", answer_raw)

    if "high" in answer:
        pred = "high"
    elif "medium" in answer:
        pred = "medium"
    elif "low" in answer:
        pred = "low"
    else:
        pred = "medium"  # fallback

    print(f"Predicción normalizada: {pred}")


if __name__ == "__main__":
    main()
