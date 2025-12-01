from src.embeddings import Embedder
from src.vectorstore import VectorStore
from src.retriever import Retriever
from src.reranker import Reranker

def main():
    question = input("Pregunta: ")
    embedder = Embedder()
    store = VectorStore()
    retriever = Retriever(embedder, store, k=15)
    reranker = Reranker()
    candidates = retriever.retrieve(question)
    top = reranker.rerank(question, candidates, top_k=5)
    for i, c in enumerate(top, 1):
        print(f"[{i}] score={c['rerank_score']:.3f} | {c['metadata']}\n{c['text'][:300]}...\n")

if __name__ == "__main__":
    main()
