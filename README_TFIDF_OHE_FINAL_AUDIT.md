# TF-IDF + OHE Baseline — Final Audit and Closure

Este documento resume el **estado final del baseline TF-IDF + OHE** tras revisar:
- el checklist técnico,
- los scripts de entrenamiento y evaluación,
- y el módulo de preparación de datos (`data_prep.py`).

Su objetivo es dejar **constancia explícita de que el baseline está cerrado**, qué decisiones están fijadas y qué aspectos quedan fuera de alcance de forma consciente.

---

## 1. Preparación de datos (data_prep.py)

Tras la revisión de `data_prep.py`, se confirma que la preparación de datos es **coherente, robusta y alineada con el resto del pipeline**.

### Normalización de la variable objetivo

- La columna `Priority` se:
  - convierte a string,
  - normaliza a minúsculas,
  - filtra para admitir únicamente `{low, medium, high}`.
- Se descartan prioridades inválidas.

Esto garantiza que:
- las clases usadas en entrenamiento y evaluación están controladas,
- la referencia a la clase `"high"` en métricas y reports es **segura y consistente**,
- no existe ambigüedad por mayúsculas, espacios o valores inesperados.

➡️ El uso de `report["high"]["recall"]` en evaluación es correcto y estable.

---

### Tratamiento de texto y metadatos

- `Body`:
  - se rellena con string vacío si es nulo,
  - se filtran textos sin palabras (`len_words > 0`).
- `Department`:
  - se rellena con `"Unknown"` si es nulo,
  - se normaliza con `strip()`.
- `Tags`:
  - se parsean como listas reales,
  - se usa su longitud como feature (`n_tags`).

Las features numéricas (`n_tags`, `len_words`) quedan correctamente definidas antes del split.

---

### Split de datos

- `train_test_split`:
  - estratificado por `Priority`,
  - `random_state` controlado por `SEED`.
- Se loggea la distribución de clases en train y test.

➡️ Esto asegura comparabilidad entre ejecuciones y consistencia en CV y test.

---

## 2. Métricas y evaluación: coherencia confirmada

Con la normalización de prioridades confirmada:

- **macro-F1** es una métrica válida y consistente para CV.
- **recall de High** es calculable de forma directa y fiable.
- `classification_report(output_dict=True)` produce claves estables (`low`, `medium`, `high`).

No existe riesgo de:
- métricas mal indexadas,
- clases faltantes en test,
- inconsistencias entre entrenamiento y evaluación.

---

## 3. Estado real del baseline TF-IDF

### Qué está completamente cerrado

- Representación:
  - TF-IDF con configuración fija (`ngram_range`, `min_df`, `sublinear_tf`).
- Metadatos:
  - `Department` (OHE / grouped / hashing evaluados),
  - `n_tags`, `len_words` (escaladas con `StandardScaler(with_mean=False)`).
- Modelos:
  - Linear SVM,
  - Logistic Regression,
  - Multinomial Naive Bayes.
- Métricas:
  - macro-F1 (principal),
  - recall High (operativa),
  - accuracy (secundaria),
  - confusion matrix en test.
- Outputs:
  - métricas guardadas en JSON,
  - matriz de confusión en PNG.

---

## 4. Punto de decisión explícito: estrategia de Department

El código **implementa correctamente** la comparación entre:
- OHE full,
- OHE agrupado,
- Hashing.

### Decisión aplicada (alineada con el código)
Se adopta la **selección automática de estrategia**:
- El pipeline elige la mejor variante según macro-F1 en CV.
- Dado que la cardinalidad es baja, los resultados son muy similares y OHE full suele ser suficiente.

Esta decisión queda fijada para evitar ambigüedad entre README y código.

## 5. Control del alcance experimental

Aunque existen utilidades opcionales para grids experimentales, el baseline se considera cerrado porque:

- la configuración TF-IDF usada para resultados finales es fija,
- no se exploran:
  - tokenizers alternativos,
  - stemming / lemmatization,
  - stopwords,
  - char n-grams,
  - feature selection,
  - tuning exhaustivo.

Estas decisiones son **conscientes** y orientadas a:
- evitar combinatoria innecesaria,
- mantener trazabilidad,
- alinear el experimento con objetivos de negocio.

---


## Diagrama de flujo (TF-IDF + OHE)

```mermaid
flowchart TD
    A[data/raw/data.csv] --> B[load() + add_features()]
    B --> C[limpieza + features n_tags/len_words]
    C --> D[split estratificado]
    D --> E[vectorizacion TF-IDF (Body)]
    D --> F[Department -> OHE/Grouped/Hash]
    D --> G[features numericas -> scaler]
    E --> H[ColumnTransformer]
    F --> H
    G --> H
    H --> I[modelo lineal: SVM / LogReg / NB]
    I --> J[CV macro-F1]
    J --> K[seleccion mejor Department]
    J --> L[seleccion mejor modelo]
    L --> M[fit modelo final]
    M --> N[metrics_test.json]
    M --> O[confusion_matrix.png]
    M --> P[models/ticket_priority.joblib]
```

## 6. Conclusión

Tras la revisión completa del pipeline y de `data_prep.py`, se concluye que:

- el baseline TF-IDF + OHE está **técnicamente completo**,
- las métricas están **bien definidas y alineadas con el objetivo de prioridad**,
- no existen frentes experimentales abiertos involuntariamente,
- el sistema es **reproducible, defendible y auditable**.

Este baseline puede considerarse **cerrado** y utilizado como referencia para enfoques más avanzados.
