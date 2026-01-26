import heapq
import json
import numpy as np
import os
from pathlib import Path
import joblib
from setfit import SetFitModel
from sklearn.preprocessing import LabelEncoder
from .config import MODEL_PATH, REPORTS_DIR
from .data_prep import load, add_features, split

SETFIT_MODEL_DIR = Path("models/setfit_model")

SLA = {"high": 4, "medium": 12, "low": 48}
SERVICE_MEAN_H = {"high": 1.5, "medium": 3.0, "low": 4.0}
SERVICE_STD_H = {"high": 0.6, "medium": 0.9, "low": 1.2}
PENALTY_EUR = {"high": 500, "medium": 200, "low": 50}
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}

rng = np.random.default_rng()


def sample_service_time(priority: str) -> float:
    mean = SERVICE_MEAN_H[priority]
    std = SERVICE_STD_H[priority]
    return max(0.25, rng.normal(mean, std))


def simulate_queue(df, priority_col: str, agents: int):
    agents = max(1, int(agents))
    agent_times = np.zeros(agents)
    completion = {}

    arrivals = df[["arrival_h", priority_col]].copy()
    arrivals["idx"] = arrivals.index
    arrivals = arrivals.sort_values("arrival_h")
    arrivals_list = arrivals[["arrival_h", priority_col, "idx"]].to_numpy().tolist()

    heap = []
    i = 0
    n = len(arrivals_list)
    current_time = 0.0

    while i < n or heap:
        next_arrival = arrivals_list[i][0] if i < n else None
        next_agent_time = float(agent_times.min())

        if heap:
            if next_arrival is None:
                current_time = max(current_time, next_agent_time)
            else:
                current_time = max(current_time, min(next_agent_time, next_arrival))
        else:
            current_time = next_arrival

        while i < n and arrivals_list[i][0] <= current_time:
            arrival_h, pr_label, idx = arrivals_list[i]
            pr = PRIORITY_ORDER.get(pr_label, 1)
            heapq.heappush(heap, (pr, arrival_h, idx))
            i += 1

        if not heap:
            continue

        for a in range(agents):
            if agent_times[a] <= current_time and heap:
                pr, arrival_h, idx = heapq.heappop(heap)
                service = df.at[idx, "service_h"]
                start = max(current_time, agent_times[a])
                finish = start + service
                agent_times[a] = finish
                completion[idx] = finish

    return df.index.to_series().map(completion)


def summarize(label: str, lead_time, sla_series):
    violations = lead_time > sla_series
    rate = violations.mean()
    mean_lead = lead_time.mean()
    mean_over = np.maximum(0, lead_time - sla_series).mean()
    print(
        f"{label}: lead_mean={mean_lead:.2f}h | viol_rate={rate:.2%} | overrun_mean={mean_over:.2f}h"
    )
    return rate, mean_over


def main():
    global rng
    seed_env = os.getenv("BUSINESS_SEED")
    seed = int(seed_env) if seed_env else None
    rng = np.random.default_rng(seed)
    split_seed = None if seed is None else seed
    seed_label = "random" if seed is None else str(seed)

    agents_env = os.getenv("BUSINESS_AGENTS")
    agents = int(agents_env) if agents_env else 8

    horizon_env = os.getenv("BUSINESS_HORIZON_H")
    horizon_h = float(horizon_env) if horizon_env else 24.0

    mode = "simple"

    print(f"[business] Split seed: {seed_label}")
    print(f"[business] Agents: {agents}")
    print(f"[business] Horizon (h): {horizon_h}")
    print(f"[business] Mode: {mode}")

    df = add_features(load())
    _, X_test, y_train, y_test = split(df, random_state=split_seed)
    print(f"[business] Test size: {len(X_test)}")
    print("[business] Test class distribution:")
    print(y_test.value_counts().to_string())

    df_test = X_test.copy()
    df_test["true_priority"] = y_test.values

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"[business] Model: {MODEL_PATH}")
        df_test["pred_priority"] = model.predict(X_test)
    else:
        model = SetFitModel.from_pretrained(SETFIT_MODEL_DIR)
        print(f"[business] Model: {SETFIT_MODEL_DIR}")
        texts = (X_test["Body"].astype(str) + " [SEP] " + X_test["Department"].astype(str)).tolist()
        df_test["pred_priority"] = model.predict(texts)

    expected = {"high", "medium", "low"}
    pred_values = set(df_test["pred_priority"].tolist())
    if not pred_values.issubset(expected):
        le = LabelEncoder()
        le.fit(y_train.astype(str))
        df_test["pred_priority"] = le.inverse_transform(
            df_test["pred_priority"].astype(int).to_numpy()
        )
        mapping = {int(i): label for i, label in enumerate(le.classes_)}
        print(f"[business] Mapped numeric labels to priorities: {mapping}")

    df_test["arrival_h"] = rng.uniform(0.0, horizon_h, len(df_test))
    df_test["service_h"] = df_test["true_priority"].apply(sample_service_time)
    df_test["sla_true"] = df_test["true_priority"].map(SLA)
    df_test["penalty_eur"] = df_test["true_priority"].map(PENALTY_EUR)

    df_test["ideal_complete_h"] = simulate_queue(df_test, "true_priority", agents)
    df_test["model_complete_h"] = simulate_queue(df_test, "pred_priority", agents)

    ideal_lead = df_test["ideal_complete_h"] - df_test["arrival_h"]
    model_lead = df_test["model_complete_h"] - df_test["arrival_h"]

    simple_sla = {"high": 4, "medium": 8, "low": 12}
    probs = df_test["true_priority"].value_counts(normalize=True)
    labels = ["high", "medium", "low"]
    p = [probs.get(lbl, 0.0) for lbl in labels]
    df_test["rand_priority"] = rng.choice(labels, size=len(df_test), p=p)

    true_h = df_test["true_priority"].map(simple_sla)
    pred_h = df_test["pred_priority"].map(simple_sla)
    rand_h = df_test["rand_priority"].map(simple_sla)

    err_model = (pred_h - true_h).abs()
    err_rand = (rand_h - true_h).abs()
    acc_model = (df_test["pred_priority"] == df_test["true_priority"]).mean()
    acc_rand = (df_test["rand_priority"] == df_test["true_priority"]).mean()

    gap_levels = [0.0, 4.0, 8.0]
    gap_model = {f"{int(g)}h": float((err_model == g).mean()) for g in gap_levels}
    gap_rand = {f"{int(g)}h": float((err_rand == g).mean()) for g in gap_levels}

    print("[business] Hour gap vs SLA targets (simple)")
    print(
        "Modelo: mean_gap=%.2fh | median_gap=%.2fh | acc=%.2f%%"
        % (err_model.mean(), err_model.median(), acc_model * 100.0)
    )
    print(
        "Aleatorio: mean_gap=%.2fh | median_gap=%.2fh | acc=%.2f%%"
        % (err_rand.mean(), err_rand.median(), acc_rand * 100.0)
    )
    print(f"Modelo: gap_0h={gap_model['0h']:.2%} | gap_4h={gap_model['4h']:.2%} | gap_8h={gap_model['8h']:.2%}")
    print(f"Aleatorio: gap_0h={gap_rand['0h']:.2%} | gap_4h={gap_rand['4h']:.2%} | gap_8h={gap_rand['8h']:.2%}")
    report = {
        "mode": mode,
        "seed": seed,
        "agents": agents,
        "horizon_h": horizon_h,
        "model_path": str(MODEL_PATH),
        "metrics": {
            "mean_gap_h": float(err_model.mean()),
            "median_gap_h": float(err_model.median()),
            "acc": float(acc_model),
            "gap_rate": gap_model,
            "baseline_mean_gap_h": float(err_rand.mean()),
            "baseline_median_gap_h": float(err_rand.median()),
            "baseline_acc": float(acc_rand),
            "baseline_gap_rate": gap_rand,
        },
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "business_metrics_simple.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[business] Report saved: {out_path}")


if __name__ == "__main__":
    main()
