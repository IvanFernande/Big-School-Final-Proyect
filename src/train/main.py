"""
Entrenamiento principal:
- TF-IDF (1-2 gram) + OneHot Department + numéricas (modelo ganador actual) → guarda modelo.
- Baseline de embeddings (config ganadora de experiments: mpnet + concat num + LogReg) para comparar métricas.
Para probar variantes (OHE agrupado, hash, grids), usar src/train/tfidf_experiments.py con RUN_EXPERIMENTS=1.
Para probar otros embeddings, usar src/train/embeddings_experiments.py.
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from src.config import MODEL_PATH, REPORTS_DIR, SEED
from src.data_prep import add_features, load, split

FAST_MODE = os.getenv("FAST_MODE", "0") == "1"
N_FOLDS = 2 if FAST_MODE else 3
NGRAM_RANGE = (1, 1) if FAST_MODE else (1, 2)
MAX_FEATURES = 8000 if FAST_MODE else None
MAX_ITER_LOGREG = 2000 if FAST_MODE else 5000
MAX_ITER_SVM = 4000 if FAST_MODE else 8000
EMB_CACHE_DIR = Path("data/processed/embeddings")
# Mejor config embeddings encontrada en experiments: mpnet base + concat num + LogReg C=2.0
EMB_MODEL = os.getenv("EMBED_MODEL_BEST", "sentence-transformers/all-mpnet-base-v2")
EMB_C = float(os.getenv("EMBED_C_BEST", "2.0"))


def build_preprocessor(ngram_range=(1, 2)):
    text_col = "Body"
    num_cols = ["n_tags", "len_words"]
    dept_cols = ["Department"]
    return ColumnTransformer(
        [
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
            ("dept", OneHotEncoder(handle_unknown="ignore"), dept_cols),
            ("num", Pipeline([("scaler", StandardScaler(with_mean=False))]), num_cols),
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


def select_best(results):
    return max(results.items(), key=lambda item: item[1]["f1_macro_mean"])


def get_embeddings(model_name: str, texts, cache_path: Path, batch_size: int = 128) -> np.ndarray:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return np.load(cache_path)
    encoder = SentenceTransformer(model_name)
    emb = np.asarray(encoder.encode(list(texts), batch_size=batch_size, show_progress_bar=True), dtype=np.float32)
    np.save(cache_path, emb)
    return emb


def train_embedding_candidate(X_train_df, X_test_df, y_train, y_test):
    print(f"[train] Embeddings baseline con {EMB_MODEL}, estrategia body_dept_concat_num, LogReg C={EMB_C}")
    emb_body_train = get_embeddings(EMB_MODEL, X_train_df["Body"], EMB_CACHE_DIR / f"{EMB_MODEL.replace('/', '_').replace(':', '_')}_body_train.npy")
    emb_body_test = get_embeddings(EMB_MODEL, X_test_df["Body"], EMB_CACHE_DIR / f"{EMB_MODEL.replace('/', '_').replace(':', '_')}_body_test.npy")
    emb_dept_train = get_embeddings(EMB_MODEL, X_train_df["Department"], EMB_CACHE_DIR / f"{EMB_MODEL.replace('/', '_').replace(':', '_')}_dept_train.npy")
    emb_dept_test = get_embeddings(EMB_MODEL, X_test_df["Department"], EMB_CACHE_DIR / f"{EMB_MODEL.replace('/', '_').replace(':', '_')}_dept_test.npy")

    scaler = StandardScaler(with_mean=False)
    num_train = scaler.fit_transform(X_train_df[["n_tags", "len_words"]])
    num_test = scaler.transform(X_test_df[["n_tags", "len_words"]])

    Xtr = np.hstack([emb_body_train, emb_dept_train, num_train])
    Xte = np.hstack([emb_body_test, emb_dept_test, num_test])

    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="saga",
        penalty="l2",
        C=EMB_C,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(clf, Xtr, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    report = classification_report(y_test, y_pred, output_dict=True)

    emb_metrics = {
        "id": f"logreg_concat_num_{EMB_MODEL.replace('/', '_')}_C{EMB_C}",
        "embedding_model": EMB_MODEL,
        "strategy": "body_dept_concat_num",
        "classifier": "logreg",
        "C": EMB_C,
        "mode": "FAST" if FAST_MODE else "FULL",
        "cv": {"f1_macro_mean": cv_scores.mean(), "f1_macro_std": cv_scores.std()},
        "test": {"classification_report": report},
    }
    with open(REPORTS_DIR / "metrics_embedding_baseline.json", "w", encoding="utf-8") as f:
        json.dump(emb_metrics, f, indent=2)
    print("[train] Métricas embedding guardadas en reports/metrics_embedding_baseline.json")
    return emb_metrics


def main():
    mode = "FAST" if FAST_MODE else "FULL"
    print(f"[train] Inicio entrenamiento (modo {mode}). N_FOLDS={N_FOLDS}, NGRAM_RANGE={NGRAM_RANGE}, MAX_FEATURES={MAX_FEATURES}")
    df = add_features(load())
    X_train, X_test, y_train, y_test = split(df)
    print(f"[train] Datos preparados. Train={len(X_train)}, Test={len(X_test)}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    models = candidate_models()
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

    # Baseline de embeddings (mejor config encontrada en experiments) para comparar
    emb_metrics = train_embedding_candidate(X_train, X_test, y_train, y_test)
    print(f"[train] Embedding baseline F1_macro cv={emb_metrics['cv']['f1_macro_mean']:.3f}, test={emb_metrics['test']['classification_report']['macro avg']['f1-score']:.3f}")

    return best_model, (X_test, y_test)


if __name__ == "__main__":
    main()
