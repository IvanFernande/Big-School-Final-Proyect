from typing import List, Dict

class SimpleGenerator:
    def __init__(self, llm):
        self.llm = llm  # placeholder for OpenAI / HF pipeline

    def answer(self, question: str, contexts: List[Dict]) -> str:
        context_text = "\n\n".join(c["text"] for c in contexts)
        prompt = f"Pregunta: {question}\nContexto:\n{context_text}\nRespuesta breve y factual:"
        return self.llm(prompt)  # adapt to your client
