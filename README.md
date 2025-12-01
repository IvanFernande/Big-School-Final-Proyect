# Asistente RAG multiformato (TFM Proyecto 2)

Base inicial alineada con el enunciado y el README_RAG de referencia.

## Estructura
- data/: documentos de 5 formatos (pdf, readme/md, csv, json, txt)
- src/: loaders, limpieza, chunking, embeddings, vector store (FAISS), retriever, reranker, generator
- scripts/: build_index.py para indexar, query.py para consultar
- notebooks/: espacio para pruebas y evaluacion
- results/: salidas y comparativas

## Pasos rapidos
1) Crea o coloca documentos en data/ por formato (usa `data/readme` o `data/txt` para Markdown/README).
2) Instala requisitos: `pip install -r requirements.txt`.
3) Construye indice: `python scripts/build_index.py`.
4) Lanza consultas: `python scripts/query.py`.

## Evaluacion
- Compara RAG basico vs RAG con reranker (top-20 -> rerank -> top-5).
- Documenta ejemplos en results/ y resume hallazgos en este README.

Completa este README con narrativa y resultados segun el enunciado.
