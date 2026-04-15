import pickle
import logging
import uvicorn
import os
import csv
from pathlib import Path
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(
    title="Iris Prediction API",
    version="1.2.0"
)

# ✅🔥 CORS FIX (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing (later replace with Netlify URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Globals
# --------------------------------------------------
model = None
iris_rows = []

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "templates" / "index.html"
DATA_FILE = BASE_DIR / "IRIS.csv"
MODEL_FILE = BASE_DIR / "model.pkl"

# --------------------------------------------------
# Auth (optional)
# --------------------------------------------------
DEFAULT_API_KEY = os.getenv("API_KEY", "Ansh")

def verify_password(
    x_api_key: Optional[str] = Header(None),
    password: Optional[str] = Header(None)
):
    provided_key = x_api_key or password
    if provided_key != DEFAULT_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# --------------------------------------------------
# ML Logic
# --------------------------------------------------
def train_model():
    df = pd.read_csv(DATA_FILE)

    X = df[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]].values
    y_map = {"setosa": 0, "versicolor": 1, "virginica": 2}
    y = df["Species"].str.lower().map(y_map).values

    model_obj = RandomForestClassifier()
    model_obj.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_obj, f)

    logging.info("Model trained")
    return model_obj


def run_prediction(features):
    pred = model.predict([features])
    return {"prediction": pred.tolist()}


# --------------------------------------------------
# Startup
# --------------------------------------------------
@app.on_event("startup")
def load():
    global model, iris_rows
    model = train_model()

    with open(DATA_FILE) as f:
        reader = csv.DictReader(f)
        iris_rows = [
            {
                "id": int(r["ID"]),
                "features": [
                    float(r["Sepal.Length"]),
                    float(r["Sepal.Width"]),
                    float(r["Petal.Length"]),
                    float(r["Petal.Width"])
                ],
                "species": r["Species"]
            }
            for r in reader
        ]

    logging.info("Startup complete")


# --------------------------------------------------
# Schema
# --------------------------------------------------
class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_length=4, max_length=4)


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def home():
    return {"message": "API running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/test-data")
def test_data():
    return {
        "summary": [{"species": r["species"], "count": 1} for r in iris_rows[:3]],
        "rows": iris_rows[:10]
    }


@app.post("/predict-ui")
def predict_ui(req: PredictionRequest):
    return run_prediction(req.features)


@app.post("/predict")
def predict(req: PredictionRequest, auth: bool = Depends(verify_password)):
    return run_prediction(req.features)


# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
