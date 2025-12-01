import numpy as np
import joblib
from .config import MODEL_PATH
from .data_prep import load, add_features, split

SLA = {"high": 4, "medium": 12, "low": 48}
rng = np.random.default_rng(0)


def simulate_resolution(priority: str) -> float:
    base = SLA[priority]
    noise = rng.normal(loc=1.2 if priority != "high" else 1.0, scale=0.4)
    return max(0.5, base * noise)


def main():
    model = joblib.load(MODEL_PATH)
    df = add_features(load())
    _, X_test, y_train, y_test = split(df)
    df_test = X_test.copy()
    df_test["true_priority"] = y_test.values
    df_test["pred_priority"] = model.predict(X_test)
    df_test["actual_res_h"] = df_test["true_priority"].apply(simulate_resolution)
    df_test["sla_true"] = df_test["true_priority"].map(SLA)
    df_test["sla_pred"] = df_test["pred_priority"].map(SLA)
    viol_true = (df_test["actual_res_h"] > df_test["sla_true"]).mean()
    viol_pred = (df_test["actual_res_h"] > df_test["sla_pred"]).mean()
    print(f"Violaciones (prioridad real): {viol_true:.2%}")
    print(f"Violaciones (usando modelo): {viol_pred:.2%}")
    cost_penalty = 200
    ahorro = (viol_true - viol_pred) * cost_penalty * len(df_test)
    print(f"Ahorro estimado: {ahorro:,.0f} EUR en el set evaluado")


if __name__ == "__main__":
    main()
