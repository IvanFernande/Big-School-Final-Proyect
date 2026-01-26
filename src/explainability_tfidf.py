"""
Explainability for TF-IDF + Linear SVM using coefficients (global) + SHAP (local).

Outputs:
- reports/figures/tfidf_global_<label>.png
- reports/explainability_tfidf/local_<idx>_<true>_<pred>.png
- reports/explainability_tfidf/summary.json

Run:
  python -m src.explainability_tfidf
"""

import json
import os
from pathlib import Path

mpl_dir = os.getenv("MPLCONFIGDIR")
if not mpl_dir:
    mpl_dir = str(Path("/tmp/mplconfig"))
    os.environ["MPLCONFIGDIR"] = mpl_dir
Path(mpl_dir).mkdir(parents=True, exist_ok=True)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.config import FIG_DIR, MODEL_PATH, REPORTS_DIR, SEED
from src.data_prep import add_features, load, split

N_GLOBAL = int(os.getenv("TFIDF_EXPLAIN_N_GLOBAL", "15"))
N_LOCAL_PER_CLASS = int(os.getenv("TFIDF_EXPLAIN_N_LOCAL_PER_CLASS", "2"))
N_BACKGROUND = int(os.getenv("TFIDF_EXPLAIN_N_BACKGROUND", "100"))
N_LOCAL_FEATURES = int(os.getenv("TFIDF_EXPLAIN_N_LOCAL_FEATURES", "10"))

EXPLAIN_DIR = REPORTS_DIR / "explainability_tfidf"


def _get_feature_names(preprocessor):
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        return None


def _clean_feature_name(name: str) -> str:
    """Normaliza prefijos del ColumnTransformer para hacer legibles los tokens."""
    if "__" not in name:
        return name
    prefix, rest = name.split("__", 1)
    if prefix == "text":
        return rest
    if prefix == "dept":
        return rest
    if prefix == "num":
        return rest
    return rest


def _plot_barh(tokens, values, title, path):
    colors = ["#4C78A8" if v >= 0 else "#F58518" for v in values]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(tokens, values, color=colors)
    ax.set_title(title)
    ax.set_xlabel("weight")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    np.random.seed(SEED)
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = add_features(load())
    X_train, X_test, y_train, y_test = split(df)

    model = joblib.load(MODEL_PATH)
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]

    X_train_vec = pre.transform(X_train)
    X_test_vec = pre.transform(X_test)
    feature_names = _get_feature_names(pre)
    if feature_names is None:
        raise RuntimeError("No se pudieron obtener los nombres de features del preprocesador.")

    classes = list(clf.classes_)
    coef = clf.coef_

    # Global: top positive tokens per class
    global_top = {}
    for class_idx, label in enumerate(classes):
        weights = coef[class_idx]
        top_idx = np.argsort(weights)[::-1][:N_GLOBAL]
        tokens = [_clean_feature_name(feature_names[i]) for i in top_idx][::-1]
        values = [float(weights[i]) for i in top_idx][::-1]
        global_top[label] = [{"token": t, "weight": v} for t, v in zip(tokens, values)]
        _plot_barh(tokens, values, f"TF-IDF Global - {label}", FIG_DIR / f"tfidf_global_{label}.png")

    # Local: SHAP explanations
    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(X_train_vec.shape[0], size=min(N_BACKGROUND, X_train_vec.shape[0]), replace=False)
    X_bg = X_train_vec[bg_idx]

    explainer = shap.LinearExplainer(clf, X_bg)
    preds = clf.predict(X_test_vec)

    local_rows = []
    for label in classes:
        idxs = np.where(y_test.values == label)[0]
        if len(idxs) == 0:
            continue
        chosen = []
        if label == "high":
            for target_pred in ("medium", "low"):
                candidates = idxs[preds[idxs] == target_pred]
                if len(candidates) > 0:
                    chosen.append(int(rng.choice(candidates, size=1)[0]))
            if len(chosen) < min(N_LOCAL_PER_CLASS, len(idxs)):
                remaining = [i for i in idxs if int(i) not in chosen]
                extra = rng.choice(remaining, size=min(N_LOCAL_PER_CLASS, len(idxs)) - len(chosen), replace=False)
                chosen.extend([int(i) for i in extra])
        else:
            chosen = rng.choice(idxs, size=min(N_LOCAL_PER_CLASS, len(idxs)), replace=False)
            chosen = [int(i) for i in chosen]
        for i in chosen:
            x_row = X_test_vec[i]
            pred_label = preds[i]
            shap_vals = explainer.shap_values(x_row)
            if isinstance(shap_vals, list):
                class_idx = classes.index(pred_label)
                sv = shap_vals[class_idx]
            else:
                sv = shap_vals

            sv = np.asarray(sv).reshape(-1)
            n_features = len(feature_names)
            if sv.size != n_features:
                if sv.size % n_features == 0:
                    sv = sv.reshape(-1, n_features)
                    sv = sv[class_idx] if sv.shape[0] > 1 else sv[0]
                else:
                    raise RuntimeError(
                        f"SHAP size mismatch: got {sv.size} values for {n_features} features."
                    )

            top_idx = np.argsort(np.abs(sv))[::-1][:N_LOCAL_FEATURES]
            tokens = [_clean_feature_name(feature_names[j]) for j in top_idx][::-1]
            values = [float(sv[j]) for j in top_idx][::-1]
            _plot_barh(tokens, values, f"TF-IDF Local - true={label} pred={pred_label}", EXPLAIN_DIR / f"local_{i}_{label}_{pred_label}.png")

            local_rows.append(
                {
                    "idx": int(i),
                    "true": str(label),
                    "pred": str(pred_label),
                    "items": [{"token": t, "value": v} for t, v in zip(tokens, values)],
                }
            )

    summary = {"global_top": global_top, "local": local_rows}
    (EXPLAIN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[explain_tfidf] Figuras globales en {FIG_DIR}")
    print(f"[explain_tfidf] Explicaciones locales en {EXPLAIN_DIR}")


if __name__ == "__main__":
    main()
