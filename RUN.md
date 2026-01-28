# Ejecutar el proyecto (guía rápida)

Esta guía resume los comandos más útiles. Todos se ejecutan desde la raíz del repo.

## 1) Preparación de datos y EDA
```
python -m src.data_prep
python -m src.eda
```

## 2) Entrenamiento principal (TF-IDF + metadatos)
```
python -m src.train
```

Modo rápido (menos folds y menos features):
```
FAST_MODE=1 python -m src.train
```

## 3) Evaluación en test (modelo TF-IDF guardado)
```
python -m src.evaluate
```

## 4) Explicabilidad TF-IDF (global + SHAP local)
```
python -m src.explainability_tfidf
```

## 5) Embeddings (experimentos + plots)
```
python -m src.train.embeddings_experiments
python -m src.plot_embeddings_strategies
python -m src.plot_embeddings_confusion
```

## 6) SetFit (entrenamiento y plots)
```
python -m src.train.deep_setfit
python -m src.plot_setfit_extras
```

## 7) LLM zero/few-shot (si se desea)
```
python -m src.train.zero_shot
python -m src.plot_llm_figures
```

## 8) Métricas de negocio
```
python -m src.business_metrics
```

## 9) Figura comparativa global
```
python -m src.plot_global_comparison
```

## 10) API de predicción (FastAPI)
```
python -m uvicorn src.serve_api:app --reload
```

Si algún script requiere un modelo entrenado (p. ej., explicabilidad), primero ejecuta el entrenamiento principal.
