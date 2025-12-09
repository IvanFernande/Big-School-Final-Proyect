# NOTES (TFM Proyecto 2)

## Objetivo
- Resolver el problema 2 del enunciado TFM con un flujo RAG local que indexa y consulta PDF/MD/TXT/CSV/JSON.
- Base minima: loaders + limpieza + chunking fijo, embeddings con Ollama (`nomic-embed-text:latest`), vectorstore FAISS y generacion con Gemini.

## Componentes clave
- Datos: carpeta `data/` con enunciado, guion, FAQ y tablas (csv/json).
- Ingestion: loaders especificos por formato + `src/preprocessing.clean_text` + `src/chunking.fixed_chunk` (chunks solapados).
- Embeddings: `src/embeddings.Embedder` llama a Ollama (`http://localhost:11434` o `OLLAMA_BASE_URL`) con `nomic-embed-text:latest` y normaliza.
- Vector store: `src/vectorstore.VectorStore` usa FAISS (IP) y guarda `index.faiss` + `data.json` (texto y metadatos alineados a cada vector).
- Recuperacion: `src/retriever.Retriever` devuelve top-k textos y metadatos para usar en query o generacion.
- Generacion: `src/generator.SimpleGenerator` con Gemini; clave en `GEMINI_API_KEY` o `config.json` (campo `gemini_api_key`).
- Evaluacion: `scripts/eval.py` lanza preguntas con substrings esperados, reintenta ante 429, puntua 0..1.

## Detalles tecnicos de ingesta (loaders)
- PDF (`src/loaders/pdf_loader.py`): PyPDF2; extrae texto por pagina, limpia y anade metadatos `source`, `page`, `type="pdf"`.
- Markdown (`src/loaders/markdown_loader.py`): lee UTF-8, conserva estructura basica, marca `type="markdown"` y `source`.
- TXT (`src/loaders/txt_loader.py`): lectura directa, `type="txt"`, util para FAQ/guias cortas.
- CSV (`src/loaders/csv_loader.py`): flattener a texto legible `columna: valor; ...`, incluye `row` y `type="csv"` para priorizar tabulares.
- JSON (`src/loaders/json_loader.py`): flatten jerarquico `clave.subclave: valor`, preserva listas con indices y marca `type="json"`.
- Todos pasan por `src/preprocessing.clean_text` y luego `src/chunking.fixed_chunk` (longitud ~400, solape 50) antes de embebido.

## Flujo actual de uso
1) Crear venv e instalar deps: `python -m venv .venv && .\.venv\Scripts\activate && pip install -r requirements.txt`.
2) Asegurar embeddings: `ollama pull nomic-embed-text:latest` (exportar `OLLAMA_BASE_URL` si no usas `localhost:11434`).
3) Colocar las fuentes en `data/` (pdf del enunciado, guion, faq, csv, json, etc.).
4) Construir indice: `python scripts/build_index.py` (crea `index/` con `index.faiss` + `data.json`).
5) Consultar: `python scripts/query.py` y escribir la pregunta (retrieve puro).
6) Evaluar/generar: `python scripts/eval.py` (requiere indice y GEMINI_API_KEY/config para generar respuestas con contexto).

## Guia para explicarlo al tutor (vision global + detalle)
- Que hace: pipeline RAG local para el problema 2; ingesta multiformato, index FAISS, retrieve y generacion opcional.
- Por que asi: embeddings locales para privacidad/control; chunking solapado para no perder contexto; prompt restrictivo para reducir alucinaciones.
- Piezas tecnicas: loaders/preprocessing/chunking -> embeddings (Ollama) -> FAISS -> Retriever -> (opcional) Gemini para respuesta.
- Demo rapida: pull modelo, indexar (`build_index.py`), preguntar (`query.py`), evaluar (`eval.py`).
- Datos y formatos: cubre PDF/MD/TXT/CSV/JSON con metadatos por source/type.
- Decisiones: `nomic-embed-text:latest` por rapidez/licencia local; FAISS IP con normalizacion; heuristica simple en `eval.py` para iterar rapido.
- Limitaciones: depende de Gemini para generar; eval por substrings; sin reranking; datos sinteticos.

## Estado y limitaciones actuales
- Embeddings con `nomic-embed-text:latest` en Ollama y FAISS para 5 formatos; chunking fijo 400/50 para dar solidez con textos largos.
- Generador por defecto via Gemini; dependemos de disponibilidad/cambios en Google AI Studio.
- Evaluacion actual es heuristica por substrings y depende de los datos sinteticos en `data/`.

## Posibles trabajos a futuro
- Modelo de generacion: permitir elegir/alternar modelos y ajustar temperatura/cotizaciones.
- Reindexar con embeddings mas solidos (ej. `all-mpnet-base-v2` u otra variante OSS) y probar su impacto en eval.
- Datos tabulares: mejorar el flattening/normalizacion de csv y json (tipos, unidades, lower case, claves jerarquicas) antes de embebido.
- Reranking: anadir reordenamiento con cross-encoder o similar para mejorar precision de los top-k.
- Evaluacion: guardar resultados en csv, metricas agregadas y pequenos reportes; ampliar bateria de preguntas y checks de completitud.
- Testing: tests unitarios para loaders, chunking y embedder; fixtures sintenticas que cubran corner cases.
- Observabilidad: logs estructurados en build/query/eval y manejo de errores mas descriptivo.
- UX: CLI que devuelva metadatos de contexto y permita ajustar k, max_contexts y modos de fallback.

## Referencias rapidas
- Config: `config.json` (puede contener `gemini_api_key`); envs `GEMINI_API_KEY`, `OLLAMA_BASE_URL`.
- Salida del indice: `index/index.faiss` (vectores) y `index/data.json` (ids, metadatos, texto de cada chunk).
- Rutas clave: `scripts/` (build/query/eval), `src/embeddings.py`, `src/vectorstore.py`, `src/retriever.py`, `src/generator.py`, `src/loaders/`.
- Prompt de generacion: en `src/generator.SimpleGenerator.answer`, restringe a contexto y pide declarar falta de datos.
- Heuristica de score: `scripts/eval.py` usa substring matching simple; SLEEP entre queries para evitar 429.
