"""
Parámetros ajustables para velocidad vs calidad:
- FAST_MODE (env var): "1" usa CV=2, ngram_range=(1,1), max_features=8000, iter más bajos; "0" usa config completa.
- N_FOLDS: número de folds en CV (por defecto 3 en modo completo, 2 en rápido).
- NGRAM_RANGE / MAX_FEATURES: rango de n-gramas y vocabulario TF-IDF (menos features = más rápido).
- MAX_ITER_LOGREG / MAX_ITER_SVM: iteraciones máximas de los clasificadores.
- RUN_PYCARET (env var): "1" ejecuta benchmark PyCaret tras entrenar; "0" lo omite.
"""

import json
import os

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from .config import MODEL_PATH, REPORTS_DIR, SEED
from .data_prep import add_features, load, split

FAST_MODE = os.getenv("FAST_MODE", "0") == "1"
N_FOLDS = 2 if FAST_MODE else 3
NGRAM_RANGE = (1, 1) if FAST_MODE else (1, 2)
MAX_FEATURES = 8000 if FAST_MODE else None
MAX_ITER_LOGREG = 2000 if FAST_MODE else 5000
MAX_ITER_SVM = 4000 if FAST_MODE else 8000


def build_preprocessor(ngram_range=(1, 2)):
    text_col = "Body"
    cat_cols = ["Department"]
    num_cols = ["n_tags", "len_words"]
    return ColumnTransformer(
        [
            ("text", TfidfVectorizer(min_df=2, ngram_range=ngram_range, max_features=MAX_FEATURES), text_col),
            ("dept", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )


def candidate_models():
    pre = build_preprocessor(ngram_range=NGRAM_RANGE)
    return {
        "logreg": Pipeline(
            [
                ("pre", pre),
                ("clf", LogisticRegression(max_iter=MAX_ITER_LOGREG, class_weight="balanced", solver="saga", penalty="l2", C=1.0, n_jobs=-1)),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("pre", pre),
                ("clf", LinearSVC(class_weight="balanced", max_iter=MAX_ITER_SVM)),
            ]
        ),
        "multinomial_nb": Pipeline(
            [
                ("pre", pre),
                ("clf", MultinomialNB()),
            ]
        ),
    }


def evaluate_candidates(models, X, y):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        results[name] = {"f1_macro_mean": scores.mean(), "f1_macro_std": scores.std()}
        print(f"{name}: F1_macro={scores.mean():.3f} (+/- {scores.std():.3f})")
    return results


def select_best(results):
    return max(results.items(), key=lambda item: item[1]["f1_macro_mean"])


def run_pycaret_benchmark(df):
    """
    Benchmark opcional con PyCaret. Se ejecuta solo si la lib esta disponible y RUN_PYCARET=1.
    Genera reports/metrics_pycaret.json con el leaderboard y los resultados del tune del mejor modelo.
    """
    try:
        from pycaret import classification as pyc
    except ImportError:
        print("PyCaret no instalado; omitiendo benchmark.")
        return None

    data = df[["Body", "Department", "n_tags", "len_words", "Priority"]].copy()
    data["Priority"] = data["Priority"].astype(str).str.lower().str.strip()

    setup_args = dict(
        data=data,
        target="Priority",
        session_id=SEED,
        fold=3,
        fold_strategy="stratifiedkfold",
        numeric_features=["n_tags", "len_words"],
        categorical_features=["Department"],
        text_features=["Body"],
        fix_imbalance=True,
        silent=True,
        verbose=False,
    )
    pyc.setup(**setup_args)

    top_models = pyc.compare_models(n_select=5, sort="F1")
    leaderboard = pyc.pull().to_dict(orient="records")
    top_model = top_models[0] if isinstance(top_models, list) else top_models
    tuned_best = pyc.tune_model(top_model, optimize="F1")
    tuned_results = pyc.pull().to_dict(orient="records")

    pycaret_metrics = {"leaderboard": leaderboard, "tuned_best_results": tuned_results}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "metrics_pycaret.json", "w", encoding="utf-8") as f:
        json.dump(pycaret_metrics, f, indent=2)

    return pycaret_metrics


def main():
    df = add_features(load())
    X_train, X_test, y_train, y_test = split(df)

    models = candidate_models()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = evaluate_candidates(models, X_train, y_train)
    with open(REPORTS_DIR / "metrics_baselines.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    best_name, _ = select_best(results)
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    print(f"Selected best model: {best_name}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    if os.getenv("RUN_PYCARET", "0") == "1":
        print("Ejecutando benchmark PyCaret...")
        run_pycaret_benchmark(df)

    return best_model, (X_test, y_test)


if __name__ == "__main__":
    main()
