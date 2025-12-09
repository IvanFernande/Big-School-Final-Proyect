# Ticket Priority Classification (TFM)

Proyecto de clasificacion de tickets (High/Medium/Low) con NLP + metadatos y capa de metricas de negocio (SLA/coste).

## Estructura
- data/raw/data.csv
- data/processed/
- src/ (pipeline de datos, entrenamiento, evaluacion, metricas de negocio, API)
- models/ (modelo serializado)
- reports/ (metrics_baselines.json, metrics_test.json) y reports/figures/ (graficos)

## Uso rapido
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m src.data_prep
python -m src.eda        # genera figuras y reporte EDA
python -m src.train
python -m src.evaluate
python -m src.business_metrics
uvicorn src.serve_api:app --reload
```
