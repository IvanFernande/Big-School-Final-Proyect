import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int = 384):
        # IndexFlatIP expects normalized vectors for cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas = []
        self.documents = []

    def add(self, ids, embeddings, metadatas, documents):
        vecs = np.asarray(embeddings, dtype=np.float32)
        if vecs.size == 0:
            return
        faiss.normalize_L2(vecs)
        if self.index.ntotal == 0:
            self.index.add(vecs)
        else:
            self.index.add(vecs)
        self.metadatas.extend(metadatas)
        self.documents.extend(documents)

    def query(self, query_embedding, k: int = 5):
        if self.index.ntotal == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        q = np.asarray([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, k)
        docs = []
        metas = []
        dists = []
        for i in idxs[0]:
            if i == -1:
                continue
            docs.append(self.documents[i])
            metas.append(self.metadatas[i])
            dists.append(float(scores[0][len(docs)-1]))
        return {"documents": [docs], "metadatas": [metas], "distances": [dists]}
