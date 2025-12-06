import os
from typing import List, Dict

import google.generativeai as genai

from src.config import get_secret


class SimpleGenerator:
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.5,
        config_path: str = "config.json",
    ):
        """
        Generador usando Gemini.
        Busca API key en env GEMINI_API_KEY o en config.json (gemini_api_key).
        """
        api_key = os.getenv("GEMINI_API_KEY") or get_secret("gemini_api_key", config_path)
        if not api_key:
            raise ValueError("Falta GEMINI_API_KEY o config.json con gemini_api_key.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = {"temperature": temperature}

    def answer(self, question: str, contexts: List[Dict], max_contexts: int = 4) -> str:
        selected = contexts[:max_contexts]
        context_text = "\n\n".join(f"- {c['text']}" for c in selected)
        prompt = (
            "Responde de forma concisa basándote solo en el contexto. "
            "Incluye números tal cual aparecen. Si no está en el contexto, di que no está disponible.\n"
            f"Pregunta: {question}\n"
            f"Contexto:\n{context_text}\n"
            "Respuesta:"
        )
        response = self.model.generate_content(prompt, generation_config=self.generation_config)
        return response.text.strip()
