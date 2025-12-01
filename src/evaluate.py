import json

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from .config import FIG_DIR, MODEL_PATH, REPORTS_DIR
from .data_prep import add_features, load, split


def main():
    model = joblib.load(MODEL_PATH)
    df = add_features(load())
    X_train, X_test, y_train, y_test = split(df)
    y_pred = model.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
