"""
Task 1 — Experiment Tracking & Model Comparison
Trains Ridge and RandomForest, logs to MLflow, saves best model.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

EXPERIMENT_NAME = "agriyield-yield-tons"
FEATURES = ["rainfall_mm", "fertilizer_kg", "field_hectares", "soil_quality"]
TARGET = "yield_tons"


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    # avoid division by zero
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"mae": round(mae, 6), "rmse": round(rmse, 6),
            "r2": round(r2, 6), "mape": round(mape, 6)}


def train():
    # ── Data ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values
    y = df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    model_configs = [
        {
            "name": "Ridge",
            "model": Ridge(alpha=1.0),
            "params": {"alpha": 1.0},
        },
        {
            "name": "RandomForest",
            "model": RandomForestRegressor(n_estimators=100, random_state=42),
            "params": {"n_estimators": 100, "random_state": 42},
        },
    ]

    results = []

    for cfg in model_configs:
        with mlflow.start_run(run_name=cfg["name"]):
            # Tag
            mlflow.set_tag("team", "ml_engineering")

            # Train
            cfg["model"].fit(X_train, y_train)
            y_pred = cfg["model"].predict(X_test)

            # Metrics
            metrics = compute_metrics(y_test, y_pred)

            # Log params & metrics
            mlflow.log_params(cfg["params"])
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("r2", metrics["r2"])
            mlflow.log_metric("mape", metrics["mape"])

            # Log model artifact
            mlflow.sklearn.log_model(cfg["model"], artifact_path="model")

            run_id = mlflow.active_run().info.run_id

        results.append({
            "name": cfg["name"],
            "model_obj": cfg["model"],
            "run_id": run_id,
            **metrics,
        })
        print(f"{cfg['name']}: MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}"
              f"  R2={metrics['r2']:.4f}  MAPE={metrics['mape']:.4f}")

    # ── Select best model by MAE ───────────────────────────────────────────────
    best = min(results, key=lambda x: x["mae"])
    print(f"\nBest model: {best['name']}  (MAE={best['mae']})")

    # Save best model to disk for api.py
    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    with open(best_model_path, "wb") as f:
        pickle.dump(best["model_obj"], f)

    # Save best model name for downstream scripts
    meta = {"best_model_name": best["name"], "best_run_id": best["run_id"],
            "best_mae": best["mae"]}
    with open(os.path.join(MODELS_DIR, "best_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Write results/step1_s1.json ───────────────────────────────────────────
    step1 = {
        "experiment_name": EXPERIMENT_NAME,
        "models": [
            {"name": r["name"], "mae": r["mae"], "rmse": r["rmse"],
             "r2": r["r2"], "mape": r["mape"]}
            for r in results
        ],
        "best_model": best["name"],
        "best_metric_name": "mae",
        "best_metric_value": best["mae"],
    }
    out_path = os.path.join(RESULTS_DIR, "step1_s1.json")
    with open(out_path, "w") as f:
        json.dump(step1, f, indent=2)
    print(f"\nSaved → {out_path}")
    return step1


if __name__ == "__main__":
    train()