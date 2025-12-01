import json

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


def build_preprocessor(ngram_range=(1, 2)):
    text_col = "Body"
    cat_cols = ["Department"]
    num_cols = ["n_tags", "len_words"]
    return ColumnTransformer(
        [
            ("text", TfidfVectorizer(min_df=2, ngram_range=ngram_range), text_col),
            ("dept", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )


def candidate_models():
    pre = build_preprocessor()
    return {
        "logreg": Pipeline(
            [
                ("pre", pre),
                ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="saga", penalty="l2", C=1.0, n_jobs=-1)),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("pre", pre),
                ("clf", LinearSVC(class_weight="balanced", max_iter=8000)),
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
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        results[name] = {"f1_macro_mean": scores.mean(), "f1_macro_std": scores.std()}
        print(f"{name}: F1_macro={scores.mean():.3f} (+/- {scores.std():.3f})")
    return results


def select_best(results):
    return max(results.items(), key=lambda item: item[1]["f1_macro_mean"])


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

    return best_model, (X_test, y_test)


if __name__ == "__main__":
    main()
