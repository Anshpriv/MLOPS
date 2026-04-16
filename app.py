import pickle
import logging
import uvicorn
import csv
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Setup
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "IRIS.csv"
MODEL_FILE = BASE_DIR / "model.pkl"

model = None
iris_rows = []

# -----------------------------
# Train Model
# -----------------------------
def train_model():
    df = pd.read_csv(DATA_FILE)

    X = df[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]].values
    y = df["Species"].astype("category").cat.codes

    clf = RandomForestClassifier()
    clf.fit(X, y)

    return clf

# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
def startup():
    global model, iris_rows

    model = train_model()

    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        iris_rows = list(reader)

# -----------------------------
# Schema
# -----------------------------
class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_length=4, max_length=4)

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/test-data")
def test_data():
    return iris_rows[:10]

@app.post("/predict-ui")
def predict_ui(request: PredictionRequest):
    try:
        features = [float(x) for x in request.features]
        prediction = model.predict([features])
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
