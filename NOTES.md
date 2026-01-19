# NOTES

## Resumen del proyecto
- Caso: clasificar tickets de soporte en High/Medium/Low con NLP + metadatos para reducir brechas de SLA y estimar ahorro.
- Flujo end-to-end en `src/`: limpieza (`data_prep.py`), EDA (`eda.py`), feature engineering y split (`data_prep.py`), entrenamiento y seleccion de modelo (`train.py`), evaluacion (`evaluate.py`), metricas de negocio (`business_metrics.py`), servicio FastAPI (`serve_api.py`).
- Dataset: `data/raw/data.csv` (~29.6k filas tras limpieza). Columnas clave: `Body` (texto), `Department` (categorica), `Tags` (lista), `Priority` (objetivo). Limpieza: se eliminan columnas tipo "Unnamed", se normaliza prioridad a lower, se rellenan vacios, se filtra prioridad fuera de {low, medium, high} y textos vacios.
- Features: TF-IDF (min_df=2, ngram_range 1-2 por defecto, `FAST_MODE` reduce a 1-gram y max_features=8000), one-hot de Department, numericas `n_tags`, `len_words`. Se asegura no fuga con `ColumnTransformer` + `Pipeline`.
- Split: 80/20 estratificado, `SEED=42`. Opcional `FAST_MODE` para iterar rapido.

## Modelado y resultados
- Modelos probados (cv estratificada, F1_macro, n_folds=3 o 2 en modo rapido): Logistic Regression (saga, class_weight balanced), Linear SVM (class_weight balanced), Multinomial NB.
- Mejor en CV: Linear SVM (F1_macro ~0.68, std ~0.016). Resultados guardados en `reports/metrics_baselines.json`. Modelo serializado en `models/ticket_priority.joblib`.
- Test set (`reports/metrics_test.json`): accuracy 0.67, F1_macro 0.66; recall High 0.73, precision High 0.69; Medium F1 0.67; Low F1 0.61.
- EDA (`reports/eda_report.json` + figuras en `reports/figures/`): desbalanceo (medium 12,126; high 11,511; low 6,013). Longitud texto p50=55 palabras (p99=175). `n_tags` p50=5. Top departamentos: Technical Support, Product Support, Customer Service. Top tags: Tech Support, IT, Performance, Feedback. Sin prioridades invalidas ni nulos relevantes.

## Metricas de negocio
- `business_metrics.py` simula resolucion segun SLA: high 4h, medium 12h, low 48h con ruido. Compara violaciones con prioridad real vs. prioridad predicha y calcula ahorro con penalizacion 200 por brecha. (Valores concretos dependen del seed; base en test set.)
- Objetivo de negocio: mejorar recall de High para reducir violaciones; cuantificar ahorro esperado y casos de uso (triar tickets automaticamente / dashboard de priorizacion).

## Despliegue y reproducibilidad
- API: `serve_api.py` con FastAPI (`/predict`) calcula features on the fly y devuelve prioridad. Requiere `models/ticket_priority.joblib` cargado con `joblib`.
- Scripts ejecutables: `python -m src.data_prep`, `python -m src.eda`, `python -m src.train`, `python -m src.evaluate`, `python -m src.business_metrics`, `uvicorn src.serve_api:app --reload`.
- Config: rutas en `src/config.py`; variables de entorno `FAST_MODE` y `RUN_PYCARET` para acelerar o lanzar benchmark opcional (PyCaret no instalado por defecto).

## Scripts y que hace cada uno
- `src/config.py`: define rutas base (datos, modelos, reportes) y semilla reproducible.
- `src/data_prep.py`: carga y limpia `data/raw/data.csv` (elimina columnas Unnamed, normaliza prioridad, rellena vacios, filtra prioridades invalidas y textos vacios), deriva `n_tags` y `len_words`, y realiza el split estratificado 80/20.
- `src/eda.py`: valida columnas, calcula resumen de calidad/balance (`reports/eda_report.json`), top palabras/bigramas por clase, y genera figuras de distribucion, correlaciones y dispersion en `reports/figures/`.
- `src/train.py`: entrypoint principal con la configuración ganadora actual (TF-IDF 1-2 gram + OHE completo de Department + numéricas). Ejecuta CV (F1_macro), guarda baselines (`reports/metrics_baselines.json`), entrena el mejor y lo serializa en `models/ticket_priority.joblib`.
- `src/train_tfidf.py`: banco de pruebas para TF-IDF/Department (OHE full vs agrupado vs hash) y suite de experimentos (`RUN_EXPERIMENTS=1`). Guarda comparativas en `reports/metrics_dept_strategies.json` y `reports/experiments/`.
- `src/evaluate.py`: carga el modelo serializado, evalua en test, imprime classification report, guarda `reports/metrics_test.json` y la matriz de confusion en `reports/figures/confusion_matrix.png`.
- `src/business_metrics.py`: simula tiempos de resolucion segun SLA, compara violaciones con prioridad real vs. predicha y estima ahorro con penalizacion fija.
- `src/serve_api.py`: API FastAPI con endpoint `/predict` que calcula features on the fly (Body, Department, len, n_tags) y devuelve prioridad usando el modelo entrenado.

## Scripts (estructura actualizada)
- `python -m src.train` → `src/train/main.py`: entrena el modelo TF-IDF ganador y calcula un baseline de embeddings (mpnet + concat num + LogReg) para comparar métricas.
- `python -m src.train.tfidf_experiments` → comparativas TF-IDF/OHE/hash y suite (`RUN_EXPERIMENTS=1`).
- `python -m src.train.embeddings_experiments` → pruebas con embeddings (Body/Department) y modelos lineales; guarda métricas en `reports/experiments/`.
- Resto igual: `src.data_prep`, `src.eda`, `src.evaluate`, `src.business_metrics`, `src.serve_api`.

## Notas tecnicas y aclaraciones
- Modos RAPIDO/COMPLETO: `FAST_MODE=1` baja folds a 2, usa solo uni-gramas y limita vocabulario (`max_features=8000`) para acelerar a costa de calidad. Por defecto se ejecuta modo completo (bi-gramas, folds=3, mas iteraciones).
- Benchmark opcional: `RUN_PYCARET=1` lanza `pycaret.classification.compare_models` y tuning; no se ejecuta si PyCaret no esta instalado.
- Desbalanceo: actualmente solo `class_weight="balanced"` en LogReg y SVM; no hay oversampling/undersampling. Pendiente probar SMOTE/RandomOverSampler y medir impacto en recall de High.
- Sin fuga de datos: `ColumnTransformer` y vectorizadores viven dentro de cada `Pipeline`, se ajustan solo con train y se validan con CV estratificada (`StratifiedKFold`).
- Tags como feature: se parsean con `ast.literal_eval` y solo se usa la longitud (`n_tags`); los valores de los tags no se codifican todavia (decision simplificada).
- Texto: no se aplica stemming/lemmatizacion ni stopwords custom (solo las de TF-IDF ingles). Pendiente evaluar mejoras de limpieza/stopwords.
- Reproducibilidad: `SEED=42`, split 80/20 estratificado, CV con folds dependiente de `FAST_MODE`.
- Metricas foco: F1_macro como principal por desbalanceo; se monitoriza recall de clase High para negocio.
- Supuestos de negocio: SLA (4/12/48h) y penalizacion 200 por brecha son simulados; el ahorro depende del seed y no usa datos reales de resolucion.
- Artefactos: modelo en `models/ticket_priority.joblib`; reportes en `reports/*.json` y figuras en `reports/figures/`. Regenerar tras cambios en datos/modelo.
- API: payload requiere `Body` y `Department`; `Tags` opcional (lista). El endpoint calcula `len_words` y `n_tags` al vuelo.

## Comparativa OHE vs hash (Department)
- Cardinalidad: 10 departamentos (mínimo 419 casos), sin categorías raras. OHE añade solo 10 columnas frente a las ~decenas de miles del TF-IDF.
- CV F1_macro (Linear SVM, TF-IDF 1-2 gram, full mode): OHE completo ≈0.706 (±0.004), OHE agrupado ≈0.706 (±0.004), hash (n_hash=2048) ≈0.706 (±0.004). `reports/metrics_dept_strategies.json` guarda el detalle.
- Justificación: no hay riesgo de explosión de dimensionalidad ni rareza; OHE completo es viable. Si en el futuro sube la cardinalidad, se puede reactivar el script con `RUN_EXPERIMENTS=1 python -m src.train` y ajustar `min_freq`/`n_hash` para medir impacto.

## Figuras
- `reports/figures/eda_priority_distribution.png`: conteo de tickets por clase (High/Medium/Low), evidencia el desbalanceo.
- `reports/figures/eda_department_top10.png`: top 10 departamentos por volumen de tickets; ayuda a identificar origenes principales.
- `reports/figures/eda_len_words_hist.png`: histograma de longitud de texto en palabras; muestra colas y mediana (~55).
- `reports/figures/eda_len_words_box.png`: distribucion de longitudes por clase para ver si alguna prioridad tiene descripciones mas largas/cortas.
- `reports/figures/eda_n_tags_hist.png`: distribucion del numero de tags por ticket; orienta sobre dispersion de metadatos.
- `reports/figures/eda_corr_numeric.png`: matriz de correlacion entre `len_words` y `n_tags`; detecta dependencia entre features numericas.
- `reports/figures/eda_len_vs_n_tags_scatter.png`: dispersion de `len_words` vs `n_tags` coloreado por prioridad; visualiza separabilidad y outliers.
- `reports/figures/confusion_matrix.png`: matriz de confusion en test del modelo seleccionado; muestra aciertos y confusiones por clase.

## Alineacion con enunciado 
- Cumplido: objetivo de negocio claro y traducido a valor (SLA/ahorro); pipeline reproducible; comparacion de modelos con CV; metricas tecnicas adecuadas; capa de negocio; servicio API; artefactos guardados en `reports/` y `models/`.
- Parcial/pending: explicabilidad (SHAP/LIME) no implementada; tuning limitado (sin busqueda de hiperparametros ni modelos adicionales); plan de despliegue/monitorizacion no detallado en README; narrativa README aun breve; visualizaciones de negocio no incluidas.

## Brechas tecnicas y siguientes mejoras
1) Explicabilidad: aplicar SHAP/LIME (texto + tabular) y documentar ejemplos (palabras/atributos que impulsan High/Medium/Low).
2) Tuning y modelos extra: Randomized/GridSearch para LogReg/SVM/NB; probar RF/GB/XGBoost/LightGBM. Medir impacto en recall High y F1_macro; actualizar `metrics_baselines.json`/`metrics_test.json`.
3) Balanceo: evaluar oversampling/undersampling (SMOTE/RandomOverSampler) vs. class_weight. Reportar cambios en confusion matrix y recall de High.
4) EDA ampliada: destacar outliers de longitud/tags, correlaciones, distribucion temporal si aplica; incluir conclusiones en README.
5) Metricas de negocio: explicitar supuestos (coste por brecha, volumen mensual), sensibilidad y escenarios "what-if" (mejora recall High +10%). Añadir graficos de SLA y ahorro.
6) Documentacion: redactar README con relato completo siguiendo el enunciado (objetivo, datos, limpieza, modelos, metricas tecnicas/negocio, explicabilidad, despliegue conceptual, conclusiones). Añadir esquema visual/textual del pipeline.
7) Despliegue conceptual: definir cadencia de retraining, monitorizacion de performance y data drift, versionado de modelos/features, estrategias de rollback.
8) Validacion adicional: ROC/AUC, matriz de confusion estilizada, ejemplo de ejecucion end-to-end (comandos y rutas de outputs).

## Estado actual para presentar
- Modelo baseline operativo con F1_macro 0.66 en test; buen recall en clase High (0.73).
- EDA y reportes generados; artefactos disponibles en `reports/` y `reports/figures/`.
- API lista para servir predicciones con el modelo entrenado.
- Metricas de negocio simuladas disponibles; falta visualizacion y sensibilidad.

---

Si quieres, lo siguiente es: generar las gráficas de negocio y/o consolidar un diagrama de pipeline, pero por contenido del enunciado ya está completo.
