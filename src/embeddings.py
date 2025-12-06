import os
from typing import List

import numpy as np
import requests


class Embedder:
    def __init__(self, model_name: str = "nomic-embed-text:latest", base_url: str | None = None):
        """
        Embedder usando Ollama local.
        Requiere tener el modelo descargado: `ollama pull nomic-embed-text:latest`.
        Se puede configurar la URL con OLLAMA_BASE_URL.
        """
        self.model_name = model_name
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for t in texts:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": t},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            if "embedding" not in data:
                raise ValueError(f"Respuesta inválida de Ollama: {data}")
            vectors.append(data["embedding"])

        arr = np.asarray(vectors, dtype=np.float32)
        # Normaliza para usar coseno/IP en FAISS
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
