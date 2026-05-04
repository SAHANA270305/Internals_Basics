"""
Task 4 — Model Promotion
• Assigns alias "champion" to version 1
• Trains a challenger with random_state=99, registers as version 2
• Promotes alias to version 2 if it has a lower MAE; otherwise keeps version 1
Saves results/step4_s7.json.
"""

import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

REGISTERED_MODEL_NAME = "agriyield-yield-tons-predictor"
EXPERIMENT_NAME = "agriyield-yield-tons"
TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"
FEATURES = ["rainfall_mm", "fertilizer_kg", "field_hectares", "soil_quality"]
TARGET = "yield_tons"
ALIAS_NAME = "champion"


def compute_mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def train_challenger(X_train, y_train, X_test, y_test, best_model_name):
    """Train challenger using the same model type as the champion but random_state=99."""
    if best_model_name == "RandomForest":
        challenger = RandomForestRegressor(n_estimators=100, random_state=99)
        params = {"n_estimators": 100, "random_state": 99}
    else:
        challenger = Ridge(alpha=1.0)
        params = {"alpha": 1.0, "random_state": 99}

    challenger.fit(X_train, y_train)
    y_pred = challenger.predict(X_test)
    mae = compute_mae(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    mask = y_test != 0
    mape = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{best_model_name}_challenger_rs99"):
        mlflow.set_tag("team", "ml_engineering")
        mlflow.set_tag("role", "challenger")
        mlflow.log_params(params)
        mlflow.log_metric("mae", round(mae, 6))
        mlflow.log_metric("rmse", round(rmse, 6))
        mlflow.log_metric("r2", round(r2, 6))
        mlflow.log_metric("mape", round(mape, 6))
        mlflow.sklearn.log_model(challenger, artifact_path="model")
        run_id = mlflow.active_run().info.run_id

    return challenger, mae, run_id


def promote():
    # ── Load champion metadata ─────────────────────────────────────────────────
    meta_path = os.path.join(MODELS_DIR, "best_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    champion_mae = meta["best_mae"]
    best_model_name = meta["best_model_name"]

    # ── Data ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values
    y = df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # ── Step 1: Load version 1 and set alias "champion" ───────────────────────
    # Version 1 was registered in Task 3; get its info
    step3_path = os.path.join(RESULTS_DIR, "step3_s6.json")
    with open(step3_path) as f:
        step3 = json.load(f)
    champion_version = step3["version"]   # should be 1

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=ALIAS_NAME,
        version=str(champion_version),
    )
    print(f"Alias '{ALIAS_NAME}' set to version {champion_version}")

    # ── Step 2: Train & register challenger ───────────────────────────────────
    _, challenger_mae, challenger_run_id = train_challenger(
        X_train, y_train, X_test, y_test, best_model_name
    )
    print(f"Challenger MAE: {challenger_mae:.6f}  (Champion MAE: {champion_mae:.6f})")

    challenger_uri = f"runs:/{challenger_run_id}/model"
    mv2 = mlflow.register_model(model_uri=challenger_uri,
                                  name=REGISTERED_MODEL_NAME)
    challenger_version = int(mv2.version)
    print(f"Challenger registered as version {challenger_version}")

    # ── Step 3: Promote if challenger is better ────────────────────────────────
    if challenger_mae < champion_mae:
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias=ALIAS_NAME,
            version=str(challenger_version),
        )
        action = "promoted"
        final_champion = challenger_version
        print(f"Challenger promoted to champion (v{challenger_version})")
    else:
        action = "kept"
        final_champion = champion_version
        print(f"Champion kept at version {champion_version}")

    # ── Write results/step4_s7.json ───────────────────────────────────────────
    step4 = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "alias_name": ALIAS_NAME,
        "champion_version": final_champion,
        "challenger_version": challenger_version,
        "action": action,
    }
    out_path = os.path.join(RESULTS_DIR, "step4_s7.json")
    with open(out_path, "w") as f:
        json.dump(step4, f, indent=2)
    print(f"Saved → {out_path}")
    return step4


if __name__ == "__main__":
    promote()