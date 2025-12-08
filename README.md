# RAG inicial (TFM Proyecto 2)

Base mÍnima para indexar y consultar documentos locales (PDF/MD/TXT/CSV/JSON). Usa FAISS + sentence-transformers.

## Estructura
- data/: coloca aquí los documentos fuente (incluye tu enunciado PDF y Guion.md si quieres indexarlos).
- src/: loaders, limpieza, chunking, embeddings y vectorstore.
- scripts/: `build_index.py` (crea índice) y `query.py` (consulta simple).

## Uso rápido
1) Crea entorno: `python -m venv .venv && .\.venv\Scripts\activate`.
2) Instala deps: `pip install -r requirements.txt`.
3) Copia documentos a `data/` (ej. `enunciado_TFM_MDATA2-.pdf`, `Guion.md`).
4) Construye índice: `python scripts/build_index.py`.
5) Consulta: `python scripts/query.py` y escribe tu pregunta.

## Configuración
- Embeddings: modelo `sentence-transformers` (por defecto `all-MiniLM-L6-v2`).
- No se almacenan claves. Si usas generación con Gemini, define `GEMINI_API_KEY` y añade el módulo de generación cuando lo necesites.

## Pendiente / siguientes pasos
- Añadir evaluación (notebooks/results) según enunciado.
- Integrar generación RAG (Gemini u otro LLM) y reranking si hace falta.
- Añadir tests básicos de loaders y chunking.

## Plan de mejora RAG (cumpliendo mínimos ejercicio 2 del enunciado)
1) Ingesta y chunking
- Tipos/metadatos en loaders (pdf/txt/md/csv/json) con `source` y `type`.
- Texto legible en CSV/JSON (columna: valor; claves jerárquicas) para dar señal al embedding.
- Chunking solapado fijo (400/50) con `max_chars` para evitar OOM.

2) Index y búsqueda
- Embeddings locales (Ollama `nomic-embed-text:latest`), FAISS IP con normalización.
- Retrieve inicial k=16 con heurística de boost por tipo (CSV para KPIs, JSON para config).
- Fallback: si la respuesta es incompleta, reintentar con k=32 y más contextos.

3) Generación con LLM
- Gemini (`gemini-2.0-flash`) vía `GEMINI_API_KEY` o `config.json`.
- Prompt restrictivo: responder solo con contexto; si no hay dato, declararlo.
- Pasar top contextos (8→12 en fallback).

4) Validación de respuesta
- Heurística de completitud: detectar “no disponible” o falta de números en preguntas numéricas.
- Si falla, ampliar contexto y reintentar; opcional reranker (cross-encoder) para más precisión.

5) Pruebas sugeridas (datos en `data/`)
- KPIs CSV: “¿SLA Q4 y tickets abiertos?”, “CSAT Q3”, “SLA Q2 y tickets cerrados”.
- Config JSON: “¿Valor del SLA objetivo?”, “¿Parámetros principales de configuración?”.
- FAQ TXT: “¿Cómo contactar con soporte y qué datos aportar?”, “¿Qué hacer si no se cumple el SLA?”.
- Manual/README: “Resume el flujo de escalado”, “Procedimientos para incidentes críticos”, “¿Qué módulos componen el sistema?”.

6) Cumplimiento mínimos TFM (ejercicio 2)
- Soporta 5 formatos (pdf/md/txt/csv/json).
- Indexación FAISS, embeddings y consulta.
- Evaluación manual de ejemplos (usar batería arriba y registrar resultados).
- Documentar configuración y uso en README.


---

# Notas del dev

- Cambiar de modelo porque Google AI Studio ha quitado algunos
- Reindexar y probar scripts/eval.py
- Ver mejoras en los datos o en el embedding del csv y json
