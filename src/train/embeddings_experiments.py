"""
Experimentos con embeddings usando sentence-transformers (requiere torch OK).
- Embeddings separados para Body y Department.
- Estrategias: concat Body+Dept, promedio Body+Dept, concat Body+Dept + numéricas.
- Modelos lineales (Linear SVM / Logistic Regression) con grid pequeño.

Ejecución:
  python -m src.train.embeddings_experiments
  FAST_MODE=1 python -m src.train.embeddings_experiments   # usa menos folds

Requisitos:
- torch y sentence-transformers instalados.
- Modelos configurables vía EMBED_MODELS (separados por coma) o usando el default MiniLM.
"""

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC

from src.config import REPORTS_DIR, SEED
from src.data_prep import add_features, load, split

FAST_MODE = os.getenv("FAST_MODE", "0") == "1"
N_FOLDS = 2 if FAST_MODE else 3
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"
EMB_CACHE_DIR = Path("data/processed/embeddings")
DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",      # rápido y ligero
    "sentence-transformers/all-mpnet-base-v2",     # más pesado, mejor calidad
]
ENV_MODELS = os.getenv("EMBED_MODELS")
EMBED_MODELS = [m.strip() for m in ENV_MODELS.split(",")] if ENV_MODELS else DEFAULT_MODELS


@dataclass
class EmbeddingConfig:
    model_name: str
    feature_strategy: str  # body_dept_concat | body_dept_avg | body_dept_concat_num
    classifier: str  # linear_svm | logreg
    C: float = 1.0


def get_embeddings(model_name: str, texts: List[str], cache_path: Path, batch_size: int = 128) -> np.ndarray:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return np.load(cache_path)
    encoder = SentenceTransformer(model_name)
    emb = np.asarray(encoder.encode(texts, batch_size=batch_size, show_progress_bar=True), dtype=np.float32)
    np.save(cache_path, emb)
    return emb


def get_num_features(df) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(with_mean=False)
    return scaler.fit_transform(df[["n_tags", "len_words"]])


def model_from_cfg(cfg: EmbeddingConfig):
    if cfg.classifier == "linear_svm":
        return LinearSVC(class_weight="balanced", C=cfg.C, max_iter=8000, dual="auto")
    if cfg.classifier == "logreg":
        return LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="saga",
            penalty="l2",
            C=cfg.C,
            n_jobs=-1,
        )
    raise ValueError(f"Clasificador no soportado: {cfg.classifier}")


def run_experiments(X_train, y_train, X_test, y_test, configs, model_name: str):
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in configs:
        print(f"[emb] {cfg}")
        clf = model_from_cfg(cfg)
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        exp_id = f"{cfg.classifier}_{cfg.feature_strategy}_{cfg.C}_{model_name.replace('/', '_').replace(':', '_')}"
        result = {
            "id": exp_id,
            "embedding_model": model_name,
            "feature_strategy": cfg.feature_strategy,
            "classifier": cfg.classifier,
            "C": cfg.C,
            "mode": "FAST" if FAST_MODE else "FULL",
            "cv": {"f1_macro_mean": cv_scores.mean(), "f1_macro_std": cv_scores.std()},
            "test": {"classification_report": report},
        }

        with open(EXPERIMENTS_DIR / f"{exp_id}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        rows.append(
            {
                "id": exp_id,
                "embedding_model": model_name,
                "feature_strategy": cfg.feature_strategy,
                "classifier": cfg.classifier,
                "C": cfg.C,
                "cv_f1_macro_mean": cv_scores.mean(),
                "cv_f1_macro_std": cv_scores.std(),
                "test_f1_macro": report.get("macro avg", {}).get("f1-score"),
                "test_recall_high": report.get("high", {}).get("recall"),
                "test_precision_high": report.get("high", {}).get("precision"),
            }
        )

    if rows:
        summary_path = EXPERIMENTS_DIR / "embeddings_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "id",
                "embedding_model",
                "feature_strategy",
                "classifier",
                "C",
                "cv_f1_macro_mean",
                "cv_f1_macro_std",
                "test_f1_macro",
                "test_recall_high",
                "test_precision_high",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[emb] Resumen guardado en {summary_path}")


def main():
    mode = "FAST" if FAST_MODE else "FULL"
    print(f"[emb] Inicio (modo {mode}). N_FOLDS={N_FOLDS}. Modelos embed: {EMBED_MODELS}")
    df = add_features(load())
    X_train_df, X_test_df, y_train, y_test = split(df)

    for emb_model in EMBED_MODELS:
        print(f"[emb] Calculando embeddings con {emb_model}")
        emb_body_train = get_embeddings(emb_model, X_train_df["Body"].tolist(), EMB_CACHE_DIR / f"{emb_model.replace('/', '_').replace(':', '_')}_body_train.npy")
        emb_body_test = get_embeddings(emb_model, X_test_df["Body"].tolist(), EMB_CACHE_DIR / f"{emb_model.replace('/', '_').replace(':', '_')}_body_test.npy")

        emb_dept_train = get_embeddings(emb_model, X_train_df["Department"].tolist(), EMB_CACHE_DIR / f"{emb_model.replace('/', '_').replace(':', '_')}_dept_train.npy")
        emb_dept_test = get_embeddings(emb_model, X_test_df["Department"].tolist(), EMB_CACHE_DIR / f"{emb_model.replace('/', '_').replace(':', '_')}_dept_test.npy")

        num_train = get_num_features(X_train_df)
        num_test = get_num_features(X_test_df)

        feature_variants = {
            "body_dept_concat": (
                np.hstack([emb_body_train, emb_dept_train]),
                np.hstack([emb_body_test, emb_dept_test]),
            ),
            "body_dept_avg": (
                (emb_body_train + emb_dept_train) / 2.0,
                (emb_body_test + emb_dept_test) / 2.0,
            ),
            "body_dept_concat_num": (
                np.hstack([emb_body_train, emb_dept_train, num_train]),
                np.hstack([emb_body_test, emb_dept_test, num_test]),
            ),
        }

        configs = []
        Cs = [0.5, 1.0, 2.0]
        for C_val in Cs:
            configs.extend(
                [
                    EmbeddingConfig(model_name=emb_model, feature_strategy="body_dept_concat", classifier="linear_svm", C=C_val),
                    EmbeddingConfig(model_name=emb_model, feature_strategy="body_dept_avg", classifier="linear_svm", C=C_val),
                    EmbeddingConfig(model_name=emb_model, feature_strategy="body_dept_concat_num", classifier="linear_svm", C=C_val),
                    EmbeddingConfig(model_name=emb_model, feature_strategy="body_dept_concat", classifier="logreg", C=C_val),
                    EmbeddingConfig(model_name=emb_model, feature_strategy="body_dept_concat_num", classifier="logreg", C=C_val),
                ]
            )

        for cfg in configs:
            Xtr, Xte = feature_variants[cfg.feature_strategy]
            run_experiments(Xtr, y_train, Xte, y_test, [cfg], emb_model)

    print("[emb] Experimentos completados.")


if __name__ == "__main__":
    main()
