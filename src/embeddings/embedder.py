"""Embedding wrapper for Ollama or sentence-transformers backends."""
import os
from typing import List, Literal

import numpy as np
import requests


class Embedder:
    def __init__(
        self,
        model_name: str = "nomic-embed-text:latest",
        base_url: str | None = None,
        backend: Literal["ollama", "sentence-transformers"] | None = None,
        st_model_name: str = "all-mpnet-base-v2",
        device: str | None = None,
    ):
        """
        Embedder con backend Ollama local (default) o sentence-transformers.
        - Ollama: requiere `ollama pull nomic-embed-text:latest`.
        - SentenceTransformers: requiere instalar `sentence-transformers`.
        """
        self.model_name = model_name
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.backend = backend or os.getenv("EMBED_BACKEND", "ollama")
        self.st_model_name = st_model_name
        self.device = device or os.getenv("EMBED_DEVICE")
        self._st_model = None

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch
        except Exception:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _encode_ollama(self, texts: List[str]) -> np.ndarray:
        """Call Ollama embeddings endpoint per text."""
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
                raise ValueError(f"Respuesta invalida de Ollama: {data}")
            vectors.append(data["embedding"])
        return np.asarray(vectors, dtype=np.float32)

    def _encode_st(self, texts: List[str]) -> np.ndarray:
        """Encode with sentence-transformers locally."""
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError("sentence-transformers no instalado.") from exc
        if self._st_model is None:
            device = self._resolve_device()
            self._st_model = SentenceTransformer(self.st_model_name, device=device)
        arr = self._st_model.encode(texts, normalize_embeddings=False)
        return np.asarray(arr, dtype=np.float32)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalized embeddings."""
        if self.backend == "sentence-transformers":
            arr = self._encode_st(texts)
        else:
            arr = self._encode_ollama(texts)
        # Normaliza para usar coseno/IP en FAISS
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
