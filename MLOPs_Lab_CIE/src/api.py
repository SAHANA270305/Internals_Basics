"""
Task 2 — FastAPI Serving
Serves the best model on port 8500.
  POST /forecast  — predict yield
  GET  /health    — health check
"""

import os
import json
import pickle
import numpy as np
import requests
import subprocess
import time
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI application (imported when run as a module / uvicorn target)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="AgriYield Forecast API")

# Load model once at startup
_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
_model = None

@app.on_event("startup")
def load_model():
    global _model
    with open(_model_path, "rb") as f:
        _model = pickle.load(f)
    print("Model loaded.")


class ForecastInput(BaseModel):
    rainfall_mm: float = Field(..., ge=200, le=1200)
    fertilizer_kg: float = Field(..., ge=50, le=300)
    field_hectares: float = Field(..., ge=1, le=20)
    soil_quality: int = Field(..., ge=1, le=5)


class ForecastResponse(BaseModel):
    prediction: float


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": _model is not None}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(data: ForecastInput):
    X = np.array([[data.rainfall_mm, data.fertilizer_kg,
                   data.field_hectares, data.soil_quality]])
    pred = float(_model.predict(X)[0])
    return ForecastResponse(prediction=round(pred, 6))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Script entry-point: start server, run test, write JSON, stop server
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_and_test():
    PORT = 8500
    TEST_INPUT = {
        "rainfall_mm": 839.7,
        "fertilizer_kg": 203.1,
        "field_hectares": 11.4,
        "soil_quality": 3,
    }
    BASE_URL = f"http://127.0.0.1:{PORT}"

    # Start uvicorn in background
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "src.api:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    for _ in range(30):
        time.sleep(1)
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
    else:
        server.terminate()
        raise RuntimeError("Server did not start in time.")

    try:
        health_resp = requests.get(f"{BASE_URL}/health").json()
        pred_resp = requests.post(f"{BASE_URL}/forecast", json=TEST_INPUT).json()
    finally:
        server.terminate()
        server.wait()

    prediction = round(pred_resp["prediction"], 6)

    step2 = {
        "health_endpoint": "/health",
        "predict_endpoint": "/forecast",
        "port": PORT,
        "health_response": health_resp,
        "test_input": TEST_INPUT,
        "prediction": prediction,
    }
    out_path = os.path.join(RESULTS_DIR, "step2_s4.json")
    with open(out_path, "w") as f:
        json.dump(step2, f, indent=2)
    print(f"Prediction: {prediction}")
    print(f"Saved → {out_path}")
    return step2


if __name__ == "__main__":
    run_and_test()