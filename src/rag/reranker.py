from typing import Dict, List


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 16,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            raise RuntimeError("sentence-transformers no disponible para reranking.") from exc
        self.model = CrossEncoder(model_name, device=device or "cpu")
        self.batch_size = batch_size

    def rerank(self, question: str, results: List[Dict], top_k: int | None = None) -> List[Dict]:
        if not results:
            return results
        pairs = [(question, r["text"]) for r in results]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        scored = []
        for r, score in zip(results, scores):
            item = dict(r)
            item["rerank_score"] = float(score)
            scored.append(item)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        if top_k:
            return scored[:top_k]
        return scored
