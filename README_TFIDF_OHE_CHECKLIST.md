# TF-IDF + OHE Baseline — Development Checklist

Este documento sirve como **guion operativo** para cerrar correctamente el baseline clasico de clasificacion con TF-IDF + metadatos, evitando abrir frentes experimentales innecesarios.
No es redaccion final, sino una **lista de verificacion tecnica**.

**Estado**
- [x] Verificado en codigo actual
- [ ] Pendiente de implementar
- [-] Opcional / no aplicable

---

## 0. Estado base (baseline minimo correcto)

Verificar que **todo lo siguiente esta implementado y no se modifica**:

### Texto
- [x] Input principal: `Body`

### TF-IDF
- [x] `ngram_range = (1, 2)` en modo FULL (`FAST_MODE` usa (1, 1) para acelerar)
- [x] `min_df = 2`
- [x] `sublinear_tf = True`
- [x] Configuracion **fija** para todos los modelos (salvo FAST_MODE)

### Metadatos
- Categorical:
  - [x] `Department` -> One-Hot Encoding (full)
- Numericos:
  - [x] `n_tags`
  - [x] `len_words`

### Modelos entrenados
- [x] Linear SVM (class_weight=balanced fijo)
- [x] Logistic Regression (class_weight=balanced fijo)
- [x] Multinomial Naive Bayes

### Validacion
- [x] Cross-validation estratificada
- [x] Mismo split / `random_state` para todos los modelos

### Metrica principal (CV)
- [x] `macro-F1`

> Si todo esto esta presente, el baseline esta correctamente construido.

---

## 1. Metricas: comprobaciones obligatorias

### 1.1 Metricas en Cross-Validation

Verificar en codigo:

- [x] `f1_score(..., average="macro")` como metrica de CV
- [x] Misma metrica para todos los modelos
- [x] No mezclar metricas entre modelos

No usar como criterio de seleccion:
- [x] weighted-F1 (puede aparecer en classification_report)
- [x] precision por clase
- [x] ROC / AUC

Objetivo:
- Comparacion justa bajo desbalance de clases

---

### 1.2 Metricas en Test (solo mejor modelo TF-IDF)

Calcular unicamente para el **mejor modelo TF-IDF**:

- [x] `accuracy`
- [x] `macro-F1`
- [x] `recall` de la clase **high** (via classification_report)

Ejemplo esperado:
```python
recall_score(y_true, y_pred, labels=["high"], average=None)
```

Notas:
- [x] No promediar High con otras clases
- [x] Recall High es metrica critica para SLA

---

## 2. Confusion Matrix (test)

Verificar / anadir:

- [x] Confusion matrix calculada **solo en test** (salida: `reports/figures/confusion_matrix.png`)
- [x] Solo para el **mejor modelo TF-IDF**
- [-] Opcional: normalizada por fila (`normalize="true"`)

No hacer:
- [x] Confusion matrices por CV
- [x] Confusion matrices por cada modelo

Objetivo:
- Analizar **tipo de error**, no solo metricas agregadas

---

## 3. Comparativa de estrategias OHE (cerrar el frente)

### 3.1 Estrategias evaluadas

- [x] OHE full
- [x] OHE agrupado (departments raros -> `Other`)
- [x] Hash (FeatureHasher)

### 3.2 Que debe existir en codigo

- [x] Un unico experimento controlado:
  - [x] Mismo modelo (mejor TF-IDF)
  - [x] Misma CV
  - [x] Metrica: `macro-F1`

Salida minima esperada (se guarda en `reports/metrics_dept_strategies.json`):
```python
{
  "ohe_full": 0.xxx,
  "ohe_grouped": 0.xxx,
  "hash": 0.xxx
}
```

Conclusion tecnica:
- [x] Dada la baja cardinalidad (~10), OHE full es suficiente

No repetir:
- [x] Por modelo
- [x] Por metrica
- [x] Por multiples seeds

---

## 4. Control de combinatoria (NO implementar)

Verificar que **NO se estan probando**:

- [x] Diferentes tokenizers
- [x] Stemming vs lemmatization
- [x] Stopwords on/off
- [x] Character n-grams
- [x] Feature selection (chi2, etc.)
- [x] Ajuste de `class_weight` (ya esta fijado a `balanced`)
- [x] Tuning exhaustivo de hiperparametros

Regla:
> Configuracion TF-IDF fija para garantizar comparabilidad.

---

## 5. Estructura minima de codigo

El proyecto debe permitir localizar facilmente:

- [x] Definicion del vectorizador TF-IDF
- [x] Preprocesado de features (texto + metadatos)
- [x] Definicion de modelos
- [x] Evaluacion CV
- [x] Evaluacion en test

No es obligatorio refactorizar, pero si:
- [x] Separacion conceptual clara

---

## 6. Outputs finales que deben existir

Antes de cerrar el baseline, comprobar que existen:

- [x] Tabla comparativa de modelos (CV, macro-F1) -> `reports/metrics_baselines.json`
- [x] Metricas en test del mejor modelo TF-IDF -> `reports/metrics_test.json`
- [x] Confusion matrix en test -> `reports/figures/confusion_matrix.png`
- [x] Comparativa OHE (full vs grouped vs hash) -> `reports/metrics_dept_strategies.json`
- [x] Eleccion explicita de:
  - [x] modelo final
  - [x] encoding final
  - [x] metricas finales

Si todo esto existe -> **baseline cerrado**.

---

## 7. Regla de oro

Antes de anadir algo nuevo, preguntarse:

> ¿Responde a una pregunta experimental ya planteada?

Si no -> **no se implementa**.

---

## 8. Proximo paso (fuera de este baseline)

Este baseline sirve como referencia para:
- Embeddings semanticos
- SetFit
- Modelos transformer
