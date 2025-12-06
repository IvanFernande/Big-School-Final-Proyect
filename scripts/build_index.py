from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders.pdf_loader import load_pdf
from src.loaders.csv_loader import load_csv
from src.loaders.json_loader import load_json
from src.loaders.txt_loader import load_txt
from src.loaders.markdown_loader import load_markdown
from src.preprocessing import clean_text
from src.chunking import fixed_chunk
from src.embeddings import Embedder
from src.vectorstore import VectorStore

ROOT = Path("data")
loaders = {
    "pdf": load_pdf,
    "csv": load_csv,
    "json": load_json,
    "txt": load_txt,
    "md": load_markdown
}


def iter_docs():
    for ext, loader in loaders.items():
        for path in ROOT.rglob(f"*.{ext}"):
            for d in loader(path):
                d["text"] = clean_text(d["text"])
                yield d


def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    embedder = Embedder()
    store = None
    batch_size = 64
    total = 0

    def chunk_iter():
        for i, doc in enumerate(iter_docs()):
            for chunk in fixed_chunk(doc):
                yield f"doc-{i}-{chunk['metadata']['start']}", chunk["text"], chunk["metadata"]

    for batch in batched(chunk_iter(), batch_size):
        ids = [b[0] for b in batch]
        texts = [b[1] for b in batch]
        mets = [b[2] for b in batch]
        embs = embedder.encode(texts)
        if store is None:
            store = VectorStore(dim=embs.shape[1])
        store.add(ids=ids, embeddings=embs, metadatas=mets, documents=texts)
        total += len(batch)

    if store is None:
        print("No se encontraron documentos en data/.")
        return

    index_dir = Path("index")
    store.save(index_dir)
    print(f"Indexados {total} chunks en {index_dir.resolve()}")


if __name__ == "__main__":
    main()
