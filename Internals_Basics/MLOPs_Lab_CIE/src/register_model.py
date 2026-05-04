"""
Task 3 — Model Versioning
Registers the best model from Task 1 into the MLflow Model Registry.
Saves results/step3_s6.json.
"""

import os
import json
import mlflow
from mlflow.tracking import MlflowClient

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

REGISTERED_MODEL_NAME = "agriyield-yield-tons-predictor"
TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"


def register():
    # ── Load best model metadata from Task 1 ──────────────────────────────────
    meta_path = os.path.join(MODELS_DIR, "best_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    run_id = meta["best_run_id"]
    best_mae = meta["best_mae"]

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # Build model URI from the logged artifact
    model_uri = f"runs:/{run_id}/model"

    # Register model (creates the registered model if it doesn't exist)
    mv = mlflow.register_model(model_uri=model_uri,
                                name=REGISTERED_MODEL_NAME)
    version = int(mv.version)
    print(f"Registered '{REGISTERED_MODEL_NAME}' as version {version}  (run_id={run_id})")

    # ── Write results/step3_s6.json ───────────────────────────────────────────
    step3 = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "version": version,
        "run_id": run_id,
        "source_metric": "mae",
        "source_metric_value": best_mae,
    }
    out_path = os.path.join(RESULTS_DIR, "step3_s6.json")
    with open(out_path, "w") as f:
        json.dump(step3, f, indent=2)
    print(f"Saved → {out_path}")
    return step3


if __name__ == "__main__":
    register()