"""
Experimentos con embeddings usando sentence-transformers (requiere torch OK).
- Embeddings para Body, Department y Tags (con cache).
- Estrategias: concat, promedio simple y ponderado, variantes con OHE + numericas.
- Modelos lineales (Linear SVM / Logistic Regression) con grid pequeño.
- Benchmark de latencia y dimension de embedding por modelo.

Ejecucion:
  python -m src.train.embeddings_experiments
  FAST_MODE=1 python -m src.train.embeddings_experiments   # usa menos folds

Requisitos:
- torch y sentence-transformers instalados.
- Modelos configurables via EMBED_MODELS (separados por coma) o usando defaults.
"""

import csv
import json
import os
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
    "sentence-transformers/all-distilroberta-v1",  # intermedio
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",  # retrieval oriented
]
ENV_MODELS = os.getenv("EMBED_MODELS")
EMBED_MODELS = [m.strip() for m in ENV_MODELS.split(",")] if ENV_MODELS else DEFAULT_MODELS

COST_MATRIX = {
    ("high", "high"): 0,
    ("high", "medium"): 8,
    ("high", "low"): 12,
    ("medium", "high"): 3,
    ("medium", "medium"): 0,
    ("medium", "low"): 1,
    ("low", "high"): 3,
    ("low", "medium"): 1,
    ("low", "low"): 0,
}


@dataclass
class EmbeddingConfig:
    model_name: str
    feature_strategy: str  # body_dept_*, body_tags_*, body_ohe_num
    classifier: str  # linear_svm | logreg
    C: float = 1.0




def tags_to_text(tags_value) -> str:
    if isinstance(tags_value, list):
        return " ".join(str(t) for t in tags_value)
    if tags_value is None:
        return ""
    return str(tags_value)

def _make_one_hot(**kwargs):
    """Crea OneHotEncoder compatible con distintas versiones de sklearn."""
    try:
        return OneHotEncoder(**kwargs, sparse_output=False)
    except TypeError:
        kwargs.pop("sparse_output", None)
        if kwargs.get("handle_unknown") == "infrequent_if_exist":
            kwargs["handle_unknown"] = "ignore"
        kwargs.pop("min_frequency", None)
        kwargs.setdefault("sparse", False)
        return OneHotEncoder(**kwargs)


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


def get_embeddings(model_name: str, texts: List[str], cache_path: Path, batch_size: int = 128):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        start = perf_counter()
        emb = np.load(cache_path)
        return emb, perf_counter() - start, True
    encoder = SentenceTransformer(model_name)
    start = perf_counter()
    emb = np.asarray(encoder.encode(texts, batch_size=batch_size, show_progress_bar=True), dtype=np.float32)
    elapsed = perf_counter() - start
    np.save(cache_path, emb)
    return emb, elapsed, False


def get_num_features(df, scaler=None):
    if scaler is None:
        scaler = StandardScaler(with_mean=False)
        return scaler.fit_transform(df[["n_tags", "len_words"]]), scaler
    return scaler.transform(df[["n_tags", "len_words"]]), scaler




def compute_cost(y_true, y_pred):
    total = 0
    for t, p in zip(y_true, y_pred):
        total += COST_MATRIX.get((t, p), 0)
    return total

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


def run_experiments(X_train, y_train, X_test, y_test, configs, model_name: str) -> list[dict]:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in configs:
        print(f"[emb] {cfg}")
        clf = model_from_cfg(cfg)
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        cv_balanced = cross_val_score(clf, X_train, y_train, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        test_balanced = balanced_accuracy_score(y_test, y_pred)
        labels = ["high", "medium", "low"]
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        conf_high_to_med = int(cm[0, 1])
        conf_high_to_low = int(cm[0, 2])

        high_idx = None
        if hasattr(clf, "classes_"):
            try:
                high_idx = list(clf.classes_).index("high")
            except ValueError:
                high_idx = None
        scores = None
        if hasattr(clf, "decision_function"):
            dec = clf.decision_function(X_test)
            if dec is not None and getattr(dec, "ndim", 0) == 2 and high_idx is not None:
                scores = dec[:, high_idx]
        if scores is None and hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test)
            if proba is not None and getattr(proba, "ndim", 0) == 2 and high_idx is not None:
                scores = proba[:, high_idx]
        high_vs_rest_ap = None
        if scores is not None:
            y_true_high = (y_test == "high").astype(int)
            high_vs_rest_ap = average_precision_score(y_true_high, scores)

        cost_total = compute_cost(list(y_test), list(y_pred))
        cost_per_ticket = cost_total / max(len(y_test), 1)

        exp_id = f"{cfg.classifier}_{cfg.feature_strategy}_{cfg.C}_{model_name.replace('/', '_').replace(':', '_')}"
        result = {
            "id": exp_id,
            "embedding_model": model_name,
            "feature_strategy": cfg.feature_strategy,
            "classifier": cfg.classifier,
            "C": cfg.C,
            "mode": "FAST" if FAST_MODE else "FULL",
            "cv": {"f1_macro_mean": cv_scores.mean(), "f1_macro_std": cv_scores.std(), "balanced_accuracy_mean": cv_balanced.mean(), "balanced_accuracy_std": cv_balanced.std()},
            "test": {"classification_report": report, "confusion_matrix": cm.tolist()},
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
                "cv_balanced_accuracy_mean": cv_balanced.mean(),
                "cv_balanced_accuracy_std": cv_balanced.std(),
                "test_f1_macro": report.get("macro avg", {}).get("f1-score"),
                "test_balanced_accuracy": test_balanced,
                "test_recall_high": report.get("high", {}).get("recall"),
                "test_precision_high": report.get("high", {}).get("precision"),
                "test_conf_high_to_med": conf_high_to_med,
                "test_conf_high_to_low": conf_high_to_low,
                "test_high_vs_rest_ap": high_vs_rest_ap,
                "test_cost_total": cost_total,
                "test_cost_per_ticket": cost_per_ticket,
            }
        )


    return rows


def main():
    mode = "FAST" if FAST_MODE else "FULL"
    print(f"[emb] Inicio (modo {mode}). N_FOLDS={N_FOLDS}. Modelos embed: {EMBED_MODELS}")
    df = add_features(load())
    X_train_df, X_test_df, y_train, y_test = split(df)

    y_train = y_train.astype(str).str.lower()
    y_test = y_test.astype(str).str.lower()

    benchmark_rows = []
    global_rows = []

    for emb_model in EMBED_MODELS:
        print(f"[emb] Calculando embeddings con {emb_model}")
        model_tag = emb_model.replace('/', '_').replace(':', '_')

        emb_body_train, t_body_train, cache_body_train = get_embeddings(
            emb_model,
            X_train_df["Body"].tolist(),
            EMB_CACHE_DIR / f"{model_tag}_body_train.npy",
        )
        emb_body_test, t_body_test, cache_body_test = get_embeddings(
            emb_model,
            X_test_df["Body"].tolist(),
            EMB_CACHE_DIR / f"{model_tag}_body_test.npy",
        )

        emb_dept_train, t_dept_train, cache_dept_train = get_embeddings(
            emb_model,
            X_train_df["Department"].tolist(),
            EMB_CACHE_DIR / f"{model_tag}_dept_train.npy",
        )
        emb_dept_test, t_dept_test, cache_dept_test = get_embeddings(
            emb_model,
            X_test_df["Department"].tolist(),
            EMB_CACHE_DIR / f"{model_tag}_dept_test.npy",
        )

        tags_train_text = df.loc[X_train_df.index, "Tags"].apply(tags_to_text).tolist()
        tags_test_text = df.loc[X_test_df.index, "Tags"].apply(tags_to_text).tolist()
        emb_tags_train, t_tags_train, cache_tags_train = get_embeddings(
            emb_model,
            tags_train_text,
            EMB_CACHE_DIR / f"{model_tag}_tags_train.npy",
        )
        emb_tags_test, t_tags_test, cache_tags_test = get_embeddings(
            emb_model,
            tags_test_text,
            EMB_CACHE_DIR / f"{model_tag}_tags_test.npy",
        )

        body_tags_train_text = (X_train_df["Body"].astype(str) + " " + df.loc[X_train_df.index, "Tags"].apply(tags_to_text)).tolist()
        body_tags_test_text = (X_test_df["Body"].astype(str) + " " + df.loc[X_test_df.index, "Tags"].apply(tags_to_text)).tolist()
        emb_body_tags_train, t_body_tags_train, cache_body_tags_train = get_embeddings(
            emb_model,
            body_tags_train_text,
            EMB_CACHE_DIR / f"{model_tag}_body_tags_train.npy",
        )
        emb_body_tags_test, t_body_tags_test, cache_body_tags_test = get_embeddings(
            emb_model,
            body_tags_test_text,
            EMB_CACHE_DIR / f"{model_tag}_body_tags_test.npy",
        )

        dim = int(emb_body_train.shape[1])
        total_time = (
            t_body_train + t_body_test + t_dept_train + t_dept_test +
            t_tags_train + t_tags_test + t_body_tags_train + t_body_tags_test
        )
        benchmark_rows.append(
            {
                "embedding_model": emb_model,
                "embedding_dim": dim,
                "time_body_train_s": round(t_body_train, 4),
                "time_body_test_s": round(t_body_test, 4),
                "time_dept_train_s": round(t_dept_train, 4),
                "time_dept_test_s": round(t_dept_test, 4),
                "time_tags_train_s": round(t_tags_train, 4),
                "time_tags_test_s": round(t_tags_test, 4),
                "time_body_tags_train_s": round(t_body_tags_train, 4),
                "time_body_tags_test_s": round(t_body_tags_test, 4),
                "time_total_s": round(total_time, 4),
                "cache_body_train": cache_body_train,
                "cache_body_test": cache_body_test,
                "cache_dept_train": cache_dept_train,
                "cache_dept_test": cache_dept_test,
                "cache_tags_train": cache_tags_train,
                "cache_tags_test": cache_tags_test,
                "cache_body_tags_train": cache_body_tags_train,
                "cache_body_tags_test": cache_body_tags_test,
            }
        )

        num_train, num_scaler = get_num_features(X_train_df)
        num_test, _ = get_num_features(X_test_df, num_scaler)

        dept_encoder = _make_one_hot(handle_unknown="ignore")
        dept_ohe_train = dept_encoder.fit_transform(X_train_df[["Department"]])
        dept_ohe_test = dept_encoder.transform(X_test_df[["Department"]])

        body_norm_train = l2_normalize(emb_body_train)
        body_norm_test = l2_normalize(emb_body_test)
        dept_norm_train = l2_normalize(emb_dept_train)
        dept_norm_test = l2_normalize(emb_dept_test)

        tags_norm_train = l2_normalize(emb_tags_train)
        tags_norm_test = l2_normalize(emb_tags_test)
        feature_variants = {
            "body_dept_concat": (
                np.hstack([emb_body_train, emb_dept_train]),
                np.hstack([emb_body_test, emb_dept_test]),
            ),
            "body_dept_avg": (
                (body_norm_train + dept_norm_train) / 2.0,
                (body_norm_test + dept_norm_test) / 2.0,
            ),
            "body_dept_avg_w30": (
                0.3 * body_norm_train + 0.7 * dept_norm_train,
                0.3 * body_norm_test + 0.7 * dept_norm_test,
            ),
            "body_dept_avg_w70": (
                0.7 * body_norm_train + 0.3 * dept_norm_train,
                0.7 * body_norm_test + 0.3 * dept_norm_test,
            ),
            "body_dept_avg_w90": (
                0.9 * body_norm_train + 0.1 * dept_norm_train,
                0.9 * body_norm_test + 0.1 * dept_norm_test,
            ),
            "body_dept_concat_num": (
                np.hstack([emb_body_train, emb_dept_train, num_train]),
                np.hstack([emb_body_test, emb_dept_test, num_test]),
            ),
            "body_ohe_num": (
                np.hstack([emb_body_train, dept_ohe_train, num_train]),
                np.hstack([emb_body_test, dept_ohe_test, num_test]),
            ),
            "body_tags": (
                emb_body_tags_train,
                emb_body_tags_test,
            ),
            "body_tags_concat": (
                np.hstack([emb_body_train, emb_tags_train]),
                np.hstack([emb_body_test, emb_tags_test]),
            ),
            "body_tags_avg": (
                (body_norm_train + tags_norm_train) / 2.0,
                (body_norm_test + tags_norm_test) / 2.0,
            ),
            "body_tags_avg_w30": (
                0.3 * body_norm_train + 0.7 * tags_norm_train,
                0.3 * body_norm_test + 0.7 * tags_norm_test,
            ),
            "body_tags_avg_w70": (
                0.7 * body_norm_train + 0.3 * tags_norm_train,
                0.7 * body_norm_test + 0.3 * tags_norm_test,
            ),
            "body_tags_avg_w90": (
                0.9 * body_norm_train + 0.1 * tags_norm_train,
                0.9 * body_norm_test + 0.1 * tags_norm_test,
            ),
            "body_tags_ohe_num": (
                np.hstack([emb_body_tags_train, dept_ohe_train, num_train]),
                np.hstack([emb_body_tags_test, dept_ohe_test, num_test]),
            ),
        }

        configs = []
        Cs = [0.5, 1.0, 2.0]
        strategies = [
            "body_dept_concat",
            "body_dept_avg",
            "body_dept_avg_w30",
            "body_dept_avg_w70",
            "body_dept_avg_w90",
            "body_dept_concat_num",
            "body_ohe_num",
            "body_tags",
            "body_tags_concat",
            "body_tags_avg",
            "body_tags_avg_w30",
            "body_tags_avg_w70",
            "body_tags_avg_w90",
            "body_tags_ohe_num",
        ]
        for C_val in Cs:
            for strat in strategies:
                configs.append(
                    EmbeddingConfig(model_name=emb_model, feature_strategy=strat, classifier="linear_svm", C=C_val)
                )
                configs.append(
                    EmbeddingConfig(model_name=emb_model, feature_strategy=strat, classifier="logreg", C=C_val)
                )

        all_rows = []
        for cfg in configs:
            Xtr, Xte = feature_variants[cfg.feature_strategy]
            rows = run_experiments(Xtr, y_train, Xte, y_test, [cfg], emb_model)
            all_rows.extend(rows)

        global_rows.extend(all_rows)

    if global_rows:
        summary_path = EXPERIMENTS_DIR / "embeddings_summary.csv"
        summary_df = pd.DataFrame(global_rows)
        sort_cols = [c for c in ["embedding_model", "feature_strategy", "classifier", "C", "cv_f1_macro_mean"] if c in summary_df.columns]
        if sort_cols:
            summary_df = summary_df.sort_values(sort_cols, ascending=[True] * len(sort_cols))
        summary_df.to_csv(summary_path, index=False)
        with open(EXPERIMENTS_DIR / "embeddings_summary.json", "w", encoding="utf-8") as f:
            json.dump(global_rows, f, ensure_ascii=False, indent=2)
        print(f"[emb] Resumen guardado en {summary_path} ({len(summary_df)} filas)")

    if benchmark_rows:
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        bench_path = EXPERIMENTS_DIR / "embeddings_benchmark.csv"
        with open(bench_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "embedding_model",
                "embedding_dim",
                "time_body_train_s",
                "time_body_test_s",
                "time_dept_train_s",
                "time_dept_test_s",
                "time_tags_train_s",
                "time_tags_test_s",
                "time_body_tags_train_s",
                "time_body_tags_test_s",
                "time_total_s",
                "cache_body_train",
                "cache_body_test",
                "cache_dept_train",
                "cache_dept_test",
                "cache_tags_train",
                "cache_tags_test",
                "cache_body_tags_train",
                "cache_body_tags_test",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(benchmark_rows)
        with open(EXPERIMENTS_DIR / "embeddings_benchmark.json", "w", encoding="utf-8") as f:
            json.dump(benchmark_rows, f, indent=2)
        print(f"[emb] Benchmark guardado en {bench_path}")

    print("[emb] Experimentos completados.")


if __name__ == "__main__":
    main()
