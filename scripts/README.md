# Scripts: uso y configuracion

Este directorio contiene los scripts de orquestacion para construir indices, ejecutar benchmarks y probar consultas.

## Requisitos basicos

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Scripts y que hacen

- `build_index.py`
  - Construye el indice principal en `index/` usando los loaders y el chunking configurado.
  - Usa embeddings segun `config.json` (backend/modelo/dispositivo).
  - Ejecutar:
    ```bash
    python scripts/build_index.py
    ```

- `query.py`
  - Consulta interactiva: recupera top‑k y genera respuesta con el LLM configurado.
  - Ejecutar:
    ```bash
    python scripts/query.py
    ```

- `eval.py`
  - Evalua el pipeline (retrieval + LLM) con el `TEST_SET`.
  - Ejecutar:
    ```bash
    python scripts/eval.py
    ```

- `benchmark_embeddings.py`
  - Compara modelos de embeddings usando retrieval vectorial y `avg_score`.
  - Ejecutar:
    ```bash
    python scripts/benchmark_embeddings.py
    ```

- `benchmark_bm25.py`
  - Grid search de BM25 (`k1`, `b`) y guarda el mejor en `config.json`.
  - Ejecutar:
    ```bash
    python scripts/benchmark_bm25.py
    ```

- `benchmark_retrieval.py`
  - Compara estrategias de retrieval (vector, TF‑IDF, BM25, hibridos, y *_rerank si esta activo).
  - Ejecutar:
    ```bash
    python scripts/benchmark_retrieval.py
    ```

- `benchmark_llm.py`
  - Compara LLMs con el retriever configurado y genera metricas finales.
  - Ejecutar:
    ```bash
    python scripts/benchmark_llm.py
    ```

## Configuracion: que cambiar y donde

Todos los parametros viven en `config.json`.

Embeddings:
- `embed_backend`, `st_model_name`, `embed_model_name`, `embed_device`

Retrieval:
- `retriever_type`, `retriever_k`, `retriever_alpha`
- `bm25_k1`, `bm25_b`

Rerank:
- `rerank_enabled`, `rerank_model_name`, `rerank_device`, `rerank_top_k`, `rerank_batch_size`

LLM:
- `llm_winner`, `llm_temperature`, `llm_max_contexts`
- `llm_benchmark_models`, `llm_benchmark_limit`
- `llm_similarity_model`, `llm_similarity_strip_punct`
- `ollama_base_url`, `gemini_api_key`

## Orden recomendado de ejecucion

1) `python scripts/build_index.py`
2) `python scripts/benchmark_embeddings.py`
3) `python scripts/benchmark_bm25.py`
4) `python scripts/benchmark_retrieval.py`
5) `python scripts/benchmark_llm.py`
