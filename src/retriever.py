from typing import List, Dict
from .embeddings import Embedder
from .vectorstore import VectorStore

class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, k: int = 5):
        self.embedder = embedder
        self.store = store
        self.k = k

    def retrieve(self, question: str) -> List[Dict]:
        q_emb = self.embedder.encode([question])[0]
        results = self.store.query(q_emb, k=self.k)
        return [
            {"text": doc, "metadata": md, "score": score}
            for doc, md, score in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ]
