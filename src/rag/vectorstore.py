import json
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np


class VectorStore:
    def __init__(
        self,
        dim: int | None = None,
        index: faiss.IndexFlatIP | None = None,
        ids: List[str] | None = None,
        metadatas: List[Dict[str, Any]] | None = None,
        documents: List[str] | None = None,
    ):
        self.index = index
        self.dim = dim or (index.d if index is not None else None)
        self.ids = ids or []
        self.metadatas = metadatas or []
        self.documents = documents or []

    def _ensure_index(self, dim: int):
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
            self.dim = dim
        elif self.index.d != dim:
            raise ValueError(f"Dim mismatch: embeddings dim {dim} != index dim {self.index.d}")

    def add(self, ids, embeddings, metadatas, documents):
        vecs = np.asarray(embeddings, dtype=np.float32)
        if vecs.size == 0:
            return
        self._ensure_index(vecs.shape[1])
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.ids.extend(ids)
        self.metadatas.extend(metadatas)
        self.documents.extend(documents)

    def query(self, query_embedding, k: int = 5):
        if self.index is None or self.index.ntotal == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "indices": [[]]}
        q = np.asarray([query_embedding], dtype=np.float32)
        if q.shape[1] != self.index.d:
            raise ValueError(f"Query dim {q.shape[1]} does not match index dim {self.index.d}")
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, k)
        docs = []
        metas = []
        dists = []
        indices = []
        for pos, i in enumerate(idxs[0]):
            if i == -1:
                continue
            docs.append(self.documents[i])
            metas.append(self.metadatas[i])
            dists.append(float(scores[0][pos]))
            indices.append(int(i))
        return {"documents": [docs], "metadatas": [metas], "distances": [dists], "indices": [indices]}

    def save(self, persist_dir: str | Path):
        if self.index is None:
            raise ValueError("No index to save")
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        index_path = persist_path / "index.faiss"
        data_path = persist_path / "data.json"
        faiss.write_index(self.index, str(index_path))
        payload = [
            {"id": id_, "metadata": md, "document": doc}
            for id_, md, doc in zip(self.ids, self.metadatas, self.documents)
        ]
        with data_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, persist_dir: str | Path):
        persist_path = Path(persist_dir)
        index_path = persist_path / "index.faiss"
        data_path = persist_path / "data.json"
        if not index_path.exists() or not data_path.exists():
            raise FileNotFoundError(f"No se encontro un indice en {persist_path}")
        index = faiss.read_index(str(index_path))
        with data_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        ids = [p["id"] for p in payload]
        metadatas = [p["metadata"] for p in payload]
        documents = [p["document"] for p in payload]
        return cls(index=index, ids=ids, metadatas=metadatas, documents=documents)
