"""
Entrenamiento y experimentos con TF-IDF + features tabulares (Department, n_tags, len_words).
Incluye comparativa de estrategias de Department (OHE completo, OHE agrupado, hash) y suite de
experimentos para dejar trazas reproducibles.
"""

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from src.config import FIG_DIR, MODEL_PATH, REPORTS_DIR, SEED
from src.data_prep import add_features, load, split

N_FOLDS = 3
NGRAM_RANGE = (1, 2)
MAX_FEATURES = None
MAX_ITER_LOGREG = 5000
MAX_ITER_SVM = 8000
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"


def _make_one_hot(**kwargs):
    """Crea OneHotEncoder compatible con distintas versiones de sklearn."""
    try:
        return OneHotEncoder(**kwargs, sparse_output=True)
    except TypeError:
        kwargs.pop("sparse_output", None)
        # Compatibilidad: versiones antiguas no soportan min_frequency ni infrequent_if_exist
        if kwargs.get("handle_unknown") == "infrequent_if_exist":
            kwargs["handle_unknown"] = "ignore"
        kwargs.pop("min_frequency", None)
        kwargs.setdefault("sparse", True)
        return OneHotEncoder(**kwargs)


def build_dept_transform(strategy: str = "ohe_full", min_freq: int = 20, n_hash: int = 2**11):
    dept_col = ["Department"]
    if strategy == "ohe_full":
        return ("dept", _make_one_hot(handle_unknown="ignore"), dept_col)
    if strategy == "ohe_grouped":
        encoder = _make_one_hot(handle_unknown="infrequent_if_exist", min_frequency=min_freq)
        return ("dept", encoder, dept_col)
    if strategy == "hash":
        hasher_pipe = Pipeline(
            [
                ("wrap", FunctionTransformer(lambda x: [[str(v)] for v in np.array(x).ravel()], validate=False)),
                ("hash", FeatureHasher(n_features=n_hash, input_type="string")),
            ]
        )
        return ("dept_hash", hasher_pipe, dept_col)
    raise ValueError(f"Estrategia de Department no soportada: {strategy}")


def build_preprocessor(ngram_range=(1, 2), dept_strategy="ohe_full", min_freq=20, n_hash=2**11):
    text_col = "Body"
    num_cols = ["n_tags", "len_words"]
    transformers = [
        (
            "text",
            TfidfVectorizer(
                min_df=2,
                ngram_range=ngram_range,
                max_features=MAX_FEATURES,
                sublinear_tf=True,
                smooth_idf=True,
            ),
            text_col,
        ),
        build_dept_transform(strategy=dept_strategy, min_freq=min_freq, n_hash=n_hash),
        ("num", Pipeline([("scaler", StandardScaler(with_mean=False))]), num_cols),
    ]
    return ColumnTransformer(transformers)


def candidate_models(dept_strategy="ohe_full", min_freq=20, n_hash=2**11):
    pre = build_preprocessor(ngram_range=NGRAM_RANGE, dept_strategy=dept_strategy, min_freq=min_freq, n_hash=n_hash)
    return {
        "logreg": Pipeline(
            [
                ("pre", pre),
                ("clf", LogisticRegression(max_iter=MAX_ITER_LOGREG, class_weight="balanced", solver="saga", C=1.0)),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("pre", pre),
                ("clf", LinearSVC(class_weight="balanced", max_iter=MAX_ITER_SVM, dual="auto")),
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
        print(f"[train] Evaluando {name} con CV={N_FOLDS}...")
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        results[name] = {"f1_macro_mean": scores.mean(), "f1_macro_std": scores.std()}
        print(f"{name}: F1_macro={scores.mean():.3f} (+/- {scores.std():.3f})")
    return results


def evaluate_dept_variants(X, y, min_freq=20, n_hash=2**11):
    variants = [
        ("ohe_full", dict(dept_strategy="ohe_full", min_freq=min_freq, n_hash=n_hash)),
        ("ohe_grouped", dict(dept_strategy="ohe_grouped", min_freq=min_freq, n_hash=n_hash)),
        ("hash", dict(dept_strategy="hash", min_freq=min_freq, n_hash=n_hash)),
    ]
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = {}
    for name, kwargs in variants:
        model = Pipeline(
            [
                ("pre", build_preprocessor(ngram_range=NGRAM_RANGE, **kwargs)),
                ("clf", LinearSVC(class_weight="balanced", max_iter=MAX_ITER_SVM, dual="auto")),
            ]
        )
        print(f"[train] Dept strategy={name}")
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        scores[name] = {"f1_macro_mean": cv_scores.mean(), "f1_macro_std": cv_scores.std()}
        print(f"{name}: F1_macro={cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    return scores


def select_best(results):
    return max(results.items(), key=lambda item: item[1]["f1_macro_mean"])


def select_best_dept_variant(scores):
    return select_best(scores)


def _model_from_name(name: str, preprocessor: ColumnTransformer, C: float = 1.0):
    if name == "linear_svm":
        return Pipeline(
            [
                ("pre", preprocessor),
                ("clf", LinearSVC(class_weight="balanced", max_iter=MAX_ITER_SVM, C=C, dual="auto")),
            ]
        )
    if name == "logreg":
        return Pipeline(
            [
                ("pre", preprocessor),
                ("clf", LogisticRegression(max_iter=MAX_ITER_LOGREG, class_weight="balanced", solver="saga", C=C)),
            ]
        )
    if name == "multinomial_nb":
        return Pipeline(
            [
                ("pre", preprocessor),
                ("clf", MultinomialNB()),
            ]
        )
    raise ValueError(f"Modelo no soportado: {name}")


def default_experiment_grid() -> List[Dict[str, Any]]:
    return [
        {"name": "svm_ohe_full", "model": "linear_svm", "dept_strategy": "ohe_full", "min_freq": 20, "n_hash": 2048, "ngram_range": NGRAM_RANGE, "C": 1.0},
        {"name": "svm_ohe_grouped20", "model": "linear_svm", "dept_strategy": "ohe_grouped", "min_freq": 20, "n_hash": 2048, "ngram_range": NGRAM_RANGE, "C": 1.0},
        {"name": "svm_ohe_grouped50", "model": "linear_svm", "dept_strategy": "ohe_grouped", "min_freq": 50, "n_hash": 2048, "ngram_range": NGRAM_RANGE, "C": 1.0},
        {"name": "svm_hash", "model": "linear_svm", "dept_strategy": "hash", "min_freq": 20, "n_hash": 2048, "ngram_range": NGRAM_RANGE, "C": 1.0},
        {"name": "logreg_ohe_grouped20", "model": "logreg", "dept_strategy": "ohe_grouped", "min_freq": 20, "n_hash": 2048, "ngram_range": NGRAM_RANGE, "C": 1.0},
    ]


def run_experiment_suite(
    X_train,
    y_train,
    X_test,
    y_test,
    grid: List[Dict[str, Any]],
    exp_dir=EXPERIMENTS_DIR,
):
    exp_dir.mkdir(parents=True, exist_ok=True)
    rows_for_csv = []

    for cfg in grid:
        name = cfg.get("name", f"exp_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}")
        model_name = cfg.get("model", "linear_svm")
        ngram_range = cfg.get("ngram_range", NGRAM_RANGE)
        dept_strategy = cfg.get("dept_strategy", "ohe_full")
        min_freq = cfg.get("min_freq", 20)
        n_hash = cfg.get("n_hash", 2**11)
        C = cfg.get("C", 1.0)

        print(f"[experiment] {name} | model={model_name}, dept={dept_strategy}, ngram={ngram_range}")
        pre = build_preprocessor(ngram_range=ngram_range, dept_strategy=dept_strategy, min_freq=min_freq, n_hash=n_hash)
        model = _model_from_name(model_name, preprocessor=pre, C=C)

        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        exp_result = {
            "id": name,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "mode": "FULL",
            "params": {
                "model": model_name,
                "C": C,
                "ngram_range": ngram_range,
                "max_features": MAX_FEATURES,
                "dept_strategy": dept_strategy,
                "min_freq": min_freq,
                "n_hash": n_hash,
                "n_folds": N_FOLDS,
            },
            "cv": {"f1_macro_mean": cv_scores.mean(), "f1_macro_std": cv_scores.std()},
            "test": {
                "classification_report": report,
            },
        }

        with open(exp_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(exp_result, f, indent=2)

        rows_for_csv.append(
            {
                "id": name,
                "model": model_name,
                "dept_strategy": dept_strategy,
                "ngram_range": str(ngram_range),
                "max_features": MAX_FEATURES,
                "min_freq": min_freq,
                "n_hash": n_hash,
                "cv_f1_macro_mean": cv_scores.mean(),
                "cv_f1_macro_std": cv_scores.std(),
                "test_f1_macro": report.get("macro avg", {}).get("f1-score"),
                "test_recall_high": report.get("high", {}).get("recall"),
                "test_precision_high": report.get("high", {}).get("precision"),
            }
        )

    summary_path = exp_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "id",
            "model",
            "dept_strategy",
            "ngram_range",
            "max_features",
            "min_freq",
            "n_hash",
            "cv_f1_macro_mean",
            "cv_f1_macro_std",
            "test_f1_macro",
            "test_recall_high",
            "test_precision_high",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_for_csv)

    print(f"[experiment] Resultados guardados en {exp_dir}")


def run_pycaret_benchmark(df):
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

    pycaret_metrics = {"leaderboard": leaderboard, "tuned_best_results": tuned_best}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "metrics_pycaret.json", "w", encoding="utf-8") as f:
        json.dump(pycaret_metrics, f, indent=2)

    return pycaret_metrics


def main():
    print(f"[train] Inicio entrenamiento (modo FULL). N_FOLDS={N_FOLDS}, NGRAM_RANGE={NGRAM_RANGE}, MAX_FEATURES={MAX_FEATURES}")
    df = add_features(load())
    X_train, X_test, y_train, y_test = split(df)
    print(f"[train] Datos preparados. Train={len(X_train)}, Test={len(X_test)}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    dept_scores = evaluate_dept_variants(X_train, y_train)
    with open(REPORTS_DIR / "metrics_dept_strategies.json", "w", encoding="utf-8") as f:
        json.dump(dept_scores, f, indent=2)
    best_dept_name, _ = select_best_dept_variant(dept_scores)
    print(f"[train] Mejor estrategia Department: {best_dept_name}")

    models = candidate_models(dept_strategy=best_dept_name)
    results = evaluate_candidates(models, X_train, y_train)
    with open(REPORTS_DIR / "metrics_baselines.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("[train] Resultados guardados en reports/metrics_baselines.json")

    best_name, _ = select_best(results)
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    print(f"Selected best model: {best_name}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"[train] Modelo guardado en {MODEL_PATH}")
    # Evaluación final en test (mismo script)
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    with open(REPORTS_DIR / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[train] Métricas guardadas en reports/metrics_test.json y matriz en reports/figures/confusion_matrix.png")


    if os.getenv("RUN_PYCARET", "0") == "1":
        print("Ejecutando benchmark PyCaret...")
        run_pycaret_benchmark(df)

    if os.getenv("RUN_EXPERIMENTS", "0") == "1":
        print("[train] Ejecutando suite de experimentos definida en default_experiment_grid()")
        run_experiment_suite(X_train, y_train, X_test, y_test, grid=default_experiment_grid())

    return best_model, (X_test, y_test)


if __name__ == "__main__":
    main()
