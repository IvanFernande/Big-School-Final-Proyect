import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from .config import FIG_DIR, REPORTS_DIR
from .data_prep import add_features, load


sns.set_theme(style="whitegrid")

REQUIRED_COLS = ["Priority", "Department", "Body", "Tags"]
NUMERIC_COLS = ["len_words", "n_tags"]
EXPECTED_PRIORITIES = {"low", "medium", "high"}

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def validate_dataframe(df):
    missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    missing_numeric = [c for c in NUMERIC_COLS if c not in df.columns]
    if missing_numeric:
        raise ValueError(f"Missing numeric feature columns: {missing_numeric}")


def save_fig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _class_balance_metrics(counts):
    if counts.empty:
        return {
            "class_balance_ratio_min_max": 1.0,
            "majority_class_pct": 0.0,
            "gini_impurity": 0.0,
            "entropy": 0.0
        }

    class_ratio = float(counts.min() / counts.max())
    total = counts.sum()
    p = counts / total

    gini = float(1 - np.sum(p ** 2))
    entropy = float(-np.sum(p * np.log2(p + 1e-12)))

    return {
        "class_balance_ratio_min_max": class_ratio,
        "majority_class_pct": float(p.max()),
        "gini_impurity": gini,
        "entropy": entropy,
    }

def build_summary(df, dropped_rows: int):
    class_counts = df["Priority"].value_counts(dropna=True)

    metrics = _class_balance_metrics(class_counts)

    len_quant = df["len_words"].quantile([0.01, 0.25, 0.5, 0.75, 0.99]).to_dict()
    tags_quant = df["n_tags"].quantile([0.01, 0.5, 0.99]).to_dict()

    dept_top = df["Department"].value_counts().head(10).to_dict()

    tags_flat = Counter(
        t for tags in df["Tags"]
        if isinstance(tags, (list, tuple))
        for t in tags
    )
    tags_top = dict(tags_flat.most_common(10))

    nulls = df.isna().sum().to_dict()
    empties = {
        "Body_empty": int((df["Body"].fillna('').str.strip() == "").sum()),
        "Department_empty": int((df["Department"].fillna('').str.strip() == "").sum()),
        "Priority_empty": int((df["Priority"].fillna('').str.strip() == "").sum()),
    }

    invalid_priority = df["Priority"].fillna('').str.lower()
    invalid_count = int(((~invalid_priority.isin(EXPECTED_PRIORITIES)) & invalid_priority.ne("")).sum())

    duplicate_ids = int(df["id"].duplicated().sum()) if "id" in df.columns else 0
    # Evitar error por columnas con listas (Tags) al calcular duplicados
    df_for_dupes = df.drop(columns=["Tags"], errors="ignore")
    duplicate_rows = int(df_for_dupes.duplicated().sum())
    zero_len = int((df["len_words"] == 0).sum())

    present = [c for c in NUMERIC_COLS if c in df.columns]
    corr = df[present].corr().round(4).to_dict() if len(present) >= 2 else {}

    col_profile = {}
    for col in df.columns:
        s = df[col]
        if s.apply(lambda v: isinstance(v, (list, tuple))).any():
            # Manejar listas/tuplas (no hashables) convirtiendo a tuplas para contar únicos
            tuples = [tuple(v) for v in s.dropna() if isinstance(v, (list, tuple))]
            n_unique = len(set(tuples))
        else:
            n_unique = int(s.nunique(dropna=True))
        col_profile[col] = {
            "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean()),
            "n_unique": n_unique,
        }

    return {
        "rows_after_drop": int(df.shape[0]),
        "rows_dropped_due_to_missing_priority": int(dropped_rows),

        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "nulls": nulls,
        "empty_strings": empties,

        "class_counts": class_counts.to_dict(),
        **metrics,

        "len_words_quantiles": len_quant,
        "n_tags_quantiles": tags_quant,
        "top_departments": dept_top,
        "top_tags": tags_top,

        "invalid_priority_count": invalid_count,
        "duplicate_id_count": duplicate_ids,
        "duplicate_rows_count": duplicate_rows,
        "zero_length_text_count": zero_len,

        "correlation_numeric": corr,
        "column_profile": col_profile,
    }

def plot_priority_distribution(df):
    df2 = df.dropna(subset=["Priority"])
    if df2.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    order = df2["Priority"].value_counts().index
    sns.countplot(data=df2, x="Priority", order=order, ax=ax)
    save_fig(fig, "eda_priority_distribution.png")


def plot_department_distribution(df):
    df2 = df.dropna(subset=["Department"])
    if df2.empty:
        return
    counts = df2["Department"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    save_fig(fig, "eda_department_top10.png")


def plot_len_words(df):
    df2 = df.dropna(subset=["len_words"])
    if df2.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df2["len_words"], bins=40, kde=True, ax=ax)
    save_fig(fig, "eda_len_words_hist.png")

    df3 = df2.dropna(subset=["Priority"])
    if df3.empty:
        return

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df3, x="Priority", y="len_words", ax=ax2)
    save_fig(fig2, "eda_len_words_box.png")


def plot_n_tags(df):
    df2 = df.dropna(subset=["n_tags"])
    if df2.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df2["n_tags"], bins=20, discrete=True, ax=ax)
    save_fig(fig, "eda_n_tags_hist.png")


def plot_correlations(df):
    present = [c for c in NUMERIC_COLS if c in df.columns]
    if len(present) < 2:
        return
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(df[present].corr(), annot=True, cmap="Blues", ax=ax, vmin=0, vmax=1)
    save_fig(fig, "eda_corr_numeric.png")


def plot_len_vs_n_tags(df):
    df2 = df.dropna(subset=["len_words", "n_tags", "Priority"])
    if df2.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df2, x="len_words", y="n_tags", hue="Priority", alpha=0.4, ax=ax)
    save_fig(fig, "eda_len_vs_n_tags_scatter.png")

def _filtered_text_df(df):
    """
    Filtro del punto 5:
    - Body no vacío
    - Body largo > 3 caracteres
    - len_words > 0
    """
    df2 = df.dropna(subset=["Body", "Priority"])
    df2 = df2[df2["len_words"] > 0]
    df2 = df2[df2["Body"].str.len() > 3]
    return df2


def compute_tfidf_top_words(df, top_k=15):
    df2 = _filtered_text_df(df)
    if df2.empty:
        return {}

    results = {}
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

    for prio in df2["Priority"].unique():
        subset = df2[df2["Priority"] == prio]["Body"]
        X = vectorizer.fit_transform(subset)
        scores = X.mean(axis=0).A1
        words = vectorizer.get_feature_names_out()
        idx = np.argsort(scores)[::-1][:top_k]
        results[prio] = [(words[i], float(scores[i])) for i in idx]

    return results


def compute_top_bigrams(df, top_k=15):
    df2 = _filtered_text_df(df)
    if df2.empty:
        return {}

    results = {}
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(2, 2), max_features=5000)

    for prio in df2["Priority"].unique():
        subset = df2[df2["Priority"] == prio]["Body"]
        X = vectorizer.fit_transform(subset)
        freqs = np.asarray(X.sum(axis=0)).ravel()
        bigs = vectorizer.get_feature_names_out()
        idx = np.argsort(freqs)[::-1][:top_k]
        results[prio] = [(bigs[i], int(freqs[i])) for i in idx]

    return results

def main():
    df = load()
    df = add_features(df)
    validate_dataframe(df)
    before = len(df)

    df = df[df["Priority"].notna() & (df["Priority"].astype(str).str.strip() != "")]

    dropped = before - len(df)
    logging.info(f"Filas eliminadas por Priority nula/vacía: {dropped}")

    summary = build_summary(df, dropped_rows=dropped)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "eda_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_priority_distribution(df)
    plot_department_distribution(df)
    plot_len_words(df)
    plot_n_tags(df)
    plot_correlations(df)
    plot_len_vs_n_tags(df)

    tfidf = compute_tfidf_top_words(df)
    bigrams = compute_top_bigrams(df)

    summary["tfidf_top_words"] = tfidf
    summary["bigrams_top"] = bigrams

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info("EDA COMPLETADO.")


if __name__ == "__main__":
    main()
