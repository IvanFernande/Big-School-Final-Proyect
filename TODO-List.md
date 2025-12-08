# Comparativa: Proyecto Ticket Priority vs Notebook Profesor ("clasificacion final.ipynb")

## 1. Contexto y objetivo
- Proyecto propio: clasificacion multiclase (High/Medium/Low) de tickets con NLP + metadatos; objetivo de negocio explicito (reducir violaciones SLA y estimar ahorro); API de servicio.
- Notebook profesor: problema tabular convertido a binario (agrupa 3->2 clases); enfoque academico con EDA profundo y comparativa de muchos modelos; sin capa de negocio ni despliegue.

## 2. Flujo y similitudes
- Pipeline ML completo: carga/limpieza -> split -> entrenamiento -> comparacion de modelos -> evaluacion.
- Modelos comunes: LogReg, Naive Bayes, SVM (profesor anade KNN, DT, RF, GB, SGD, LightGBM via PyCaret).
- Balanceo: profesor usa `RandomOverSampler`; proyecto usa `class_weight` en LogReg/SVM.
- Transformaciones en pipeline para evitar fugas.

## 3. Diferencias clave
- Datos/target: proyecto = texto + metadatos, 3 clases; profesor = tabular, binario.
- Ingenieria de variables: proyecto usa TF-IDF, one-hot dept, contadores (`n_tags`, `len_words`); profesor aplica escalado, normalizacion, seleccion por varianza, PCA 2D, analisis de outliers.
- Modelado: proyecto compara 3 modelos con CV estratificada; profesor explora ~8 modelos + AutoML (PyCaret) con RandomizedSearch y visualizaciones de fronteras/ROC/CM.
- Metricas de negocio: solo en el proyecto (SLA/ahorro); el notebook solo metricas tecnicas.
- Explicabilidad: ninguno implementa SHAP/LIME; proyecto lo contempla en requirements; notebook se centra en matrices de confusion y ROC.
- Despliegue: proyecto tiene serializacion y API FastAPI; notebook no aborda despliegue ni MLOps.

## 4. Alineacion con enunciado (Proyecto 1: "Modelizacion Predictiva y Metricas de Negocio")
- Objetivo de negocio: proyecto si; notebook no.
- EDA y preparacion: notebook mas profundo (outliers, balanceo, PCA, seleccion de varianza); proyecto con EDA minimo en codigo (se puede ampliar en notebooks).
- Comparacion de modelos: ambos; notebook mas amplio y con tuning.
- Explicabilidad: pendiente en ambos; necesario anadir SHAP/LIME o feature importance/PDP.
- Metricas de negocio: cubiertas en el proyecto; ausentes en el notebook.
- Plan de despliegue: cubierto conceptualmente por API; no cubierto en notebook.
- Narrativa README: proyecto ya tiene; notebook es exploratorio sin relato de negocio.

## 5. Que aprovechar del notebook
- Profundizar EDA: distribuciones, outliers (longitud de texto), correlaciones de metadatos.
- Balanceo: probar `RandomOverSampler`/`class_weight` y medir impacto en recall de High.
- Hiperparametros: incorporar `RandomizedSearchCV`/`GridSearchCV` y, si hay tiempo, anadir RF/GB/LightGBM.
- PCA/visualizaciones: usar reducciones (PCA/UMAP sobre embeddings) para ilustrar separabilidad y soportar narrativa.
- Metricas: anadir ROC/AUC y matrices de confusion estilizadas como en el notebook.

## 6. Donde el proyecto ya supera al notebook
- Objetivo y metricas de negocio (SLA/ahorro).
- Pipeline modular reproducible, serializacion y API.
- Caso realista NLP multiclase (priorizacion de soporte).

## 7. Brechas a cerrar para cumplir al 100% el enunciado
- Implementar explicabilidad (SHAP/LIME) y documentar ejemplos.
- Ampliar EDA documentado (notebook `01_eda.ipynb`): limpieza, outliers, balanceo, decisiones.
- Anadir plan de produccion/monitorizacion en README: drift, retraining, logging, alertas.
- Ampliar o justificar la seleccion de modelos; reportar tuning y eleccion final.
- Integrar metricas de negocio en el README con cifras y supuestos claros.

## 8. Proximos pasos sugeridos
1) Crear script de EDA (texto + metadatos) y guardar figuras en `reports/figures`.
2) Anadir pipeline de balanceo y busqueda de hiperparametros para los 3 modelos actuales; opcional: RF/GB/LightGBM.
3) Implementar SHAP (tabular + texto) o LIME y documentar resultados clave.
4) Actualizar `README.md` con narrativa de negocio, metricas tecnicas y de negocio, y plan de despliegue/monitorizacion.
5) Mantener API servible con el mejor modelo y versionar artefactos en `models/`.

## Plan de trabajo propuesto
- EDA y limpieza: crear script (por ejemplo `src/eda.py`) con distribuciones, outliers (longitud texto), balanceo de clases y correlaciones de metadatos; guardar figuras en `reports/figures`.
- Balanceo y tuning: probar `RandomOverSampler` y `class_weight`; `RandomizedSearchCV` para LogReg/SVM/NB y evaluar RF/GB/LightGBM; reportar impacto en recall de High y F1 macro.
- Explicabilidad: aplicar SHAP/LIME al mejor modelo (texto + metadatos), capturar figuras y ejemplos; documentar hallazgos clave.
- Evaluacion tecnica y negocio: consolidar `metrics_baselines.json` y `metrics_test.json`, anadir ROC/AUC y matriz de confusion estilizada; refinar `business_metrics` con supuestos claros y sensibilidad de ahorro.
- Documentacion y despliegue: actualizar `README.md` con narrativa end-to-end (objetivo de negocio, datos, EDA, modelos, tuning, explicabilidad, metricas tecnicas/negocio, conclusiones) y plan de produccion (API, versionado de modelo, retraining/monitorizacion).
- Futuro: extender `src/eda.py` para autocorreccion opcional (normalizar prioridades, imputar/capar outliers en longitudes/tags, reemplazar vacios) antes de entrenamiento.