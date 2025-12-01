from pathlib import Path
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
    "md": load_markdown,
    "markdown": load_markdown,
    "readme": load_markdown,
}

def collect_docs():
    docs = []
    for ext, loader in loaders.items():
        for path in ROOT.rglob(f"*.{ext}"):
            for d in loader(path):
                d["text"] = clean_text(d["text"])
                docs.append(d)
    return docs

def main():
    docs = collect_docs()
    embedder = Embedder()
    store = VectorStore()
    ids, mets, texts = [], [], []
    for i, doc in enumerate(docs):
        for chunk in fixed_chunk(doc):
            ids.append(f"doc-{i}-{chunk['start']}")
            texts.append(chunk["text"])
            mets.append(chunk["metadata"])
    embs = embedder.encode(texts)
    store.add(ids=ids, embeddings=embs, metadatas=mets, documents=texts)
    print(f"Indexados {len(ids)} chunks")

if __name__ == "__main__":
    main()
