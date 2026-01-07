from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.embeddings import Embedder
from src.rag.vectorstore import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, k: int = 8):
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


class KeywordRetriever:
    """Keyword search (TF-IDF) sin embeddings."""

    def __init__(self, store: VectorStore, k: int = 8):
        self.store = store
        self.k = k
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(store.documents)

    def retrieve(self, question: str) -> List[Dict]:
        q_tfidf = self.vectorizer.transform([question])
        scores = (q_tfidf @ self.tfidf_matrix.T).toarray()[0]
        top_idx = np.argsort(scores)[::-1][: self.k]
        results = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue
            results.append(
                {
                    "text": self.store.documents[int(idx)],
                    "metadata": self.store.metadatas[int(idx)],
                    "score": float(scores[idx]),
                }
            )
        return results


class HybridRetriever:
    """Combina embeddings + keyword (TF-IDF) con score mixto."""

    def __init__(self, embedder: Embedder, store: VectorStore, k: int = 8, alpha: float = 0.6):
        self.embedder = embedder
        self.store = store
        self.k = k
        self.alpha = alpha
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(store.documents)

    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}
        values = np.array(list(scores.values()), dtype=np.float32)
        min_v = float(values.min())
        max_v = float(values.max())
        if max_v - min_v == 0:
            return {k: 0.0 for k in scores}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}

    def retrieve(self, question: str) -> List[Dict]:
        q_emb = self.embedder.encode([question])[0]
        k_embed = max(self.k * 4, 20)
        emb_results = self.store.query(q_emb, k=k_embed)
        emb_scores = {
            idx: score
            for idx, score in zip(emb_results["indices"][0], emb_results["distances"][0])
        }

        k_kw = max(self.k * 4, 20)
        q_tfidf = self.vectorizer.transform([question])
        kw_scores_full = (q_tfidf @ self.tfidf_matrix.T).toarray()[0]
        kw_top_idx = np.argsort(kw_scores_full)[::-1][:k_kw]
        kw_scores = {int(i): float(kw_scores_full[i]) for i in kw_top_idx if kw_scores_full[i] > 0}

        emb_norm = self._normalize_scores(emb_scores)
        kw_norm = self._normalize_scores(kw_scores)
        candidates = set(emb_norm.keys()) | set(kw_norm.keys())
        combined = {}
        for idx in candidates:
            combined[idx] = self.alpha * emb_norm.get(idx, 0.0) + (1 - self.alpha) * kw_norm.get(idx, 0.0)

        top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[: self.k]
        results = []
        for idx, score in top:
            results.append(
                {"text": self.store.documents[idx], "metadata": self.store.metadatas[idx], "score": float(score)}
            )
        return results
