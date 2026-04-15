import pickle
import logging
import uvicorn
import os
import csv
from pathlib import Path
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------
# FastAPI App Initialization
# --------------------------------------------------
app = FastAPI(
    title="Iris Prediction API",
    description="API and frontend for iris flower species prediction",
    version="1.2.0"
)

# --------------------------------------------------
# Global Model Variable
# --------------------------------------------------
model = None
iris_rows = []
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "templates" / "index.html"
DATA_FILE = BASE_DIR / "IRIS.csv"
MODEL_FILE = BASE_DIR / "model.pkl"
DEFAULT_API_KEY = os.getenv("API_KEY", "Ansh")

# --------------------------------------------------
# Simple Header Authentication Dependency
# --------------------------------------------------
def verify_password(
    x_api_key: Optional[str] = Header(None),
    password: Optional[str] = Header(None)
):
    provided_key = x_api_key or password

    if provided_key != DEFAULT_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing API key header"
        )
    return True


def run_prediction(features: List[float]):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded. Please contact the administrator."
        )

    if not features:
        raise HTTPException(
            status_code=422,
            detail="Feature list cannot be empty."
        )

    try:
        prediction = model.predict([features])
        return {
            "success": True,
            "prediction": prediction.tolist()
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input values: {str(e)}"
        )

    except TypeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Type error during prediction: {str(e)}"
        )

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction."
        )


def format_species_name(value: str) -> str:
    return value.strip().capitalize()


def label_prediction(value):
    mapping = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica",
        "0": "Setosa",
        "1": "Versicolor",
        "2": "Virginica"
    }
    return mapping.get(value, format_species_name(str(value)))


def train_model() -> object:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    expected_columns = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species"}
    if not expected_columns.issubset(df.columns):
        raise ValueError("IRIS.csv is missing required feature or label columns.")

    df["Species"] = df["Species"].astype(str).str.strip().str.lower()
    label_map = {"setosa": 0, "versicolor": 1, "virginica": 2}
    if not set(df["Species"]).issubset(set(label_map)):
        raise ValueError("IRIS.csv contains unknown species labels.")

    X = df[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]].astype(float).values
    y = df["Species"].map(label_map).astype(int).values

    model_obj = RandomForestClassifier(n_estimators=100, random_state=42)
    model_obj.fit(X, y)

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(model_obj, file)

    logging.info("Trained new model from IRIS.csv and saved to model.pkl")
    return model_obj


def build_test_data_payload(limit: int = 12):
    if not iris_rows:
        return {
            "summary": [],
            "rows": []
        }

    species_counts = {}
    preview_rows = []

    for row in iris_rows:
        species = row["species"]
        species_counts[species] = species_counts.get(species, 0) + 1

    for row in iris_rows[:limit]:
        features = row["features"]
        predicted_value = run_prediction(features)["prediction"][0]
        preview_rows.append(
            {
                "id": row["id"],
                "features": features,
                "actual_species": row["species"],
                "predicted_species": label_prediction(predicted_value)
            }
        )

    return {
        "summary": [
            {"species": species, "count": count}
            for species, count in species_counts.items()
        ],
        "rows": preview_rows
    }

# --------------------------------------------------
# Startup Event – Load Model
# --------------------------------------------------
@app.on_event("startup")
def load_model():
    global model, iris_rows
    try:
        model = train_model()

        with open(DATA_FILE, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            iris_rows = [
                {
                    "id": int(row["ID"]),
                    "features": [
                        float(row["Sepal.Length"]),
                        float(row["Sepal.Width"]),
                        float(row["Petal.Length"]),
                        float(row["Petal.Width"])
                    ],
                    "species": format_species_name(row["Species"])
                }
                for row in reader
            ]

        logging.info("Model trained and dataset loaded successfully.")

    except FileNotFoundError as e:
        logging.error(f"Required file not found: {e}")
        model = None
        iris_rows = []

    except Exception as e:
        logging.error(f"Unexpected error during startup: {e}")
        model = None
        iris_rows = []


# --------------------------------------------------
# Request Schema
# --------------------------------------------------
class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="List of four iris flower measurements"
    )


# --------------------------------------------------
# Frontend Route
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    try:
        return HTMLResponse(INDEX_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Frontend file not found.")


# --------------------------------------------------
# Health Endpoint
# --------------------------------------------------
@app.get("/health")
def health():
    return JSONResponse(
        {
            "status": "ok",
            "model_loaded": model is not None
        }
    )


@app.get("/test-data")
def test_data():
    return JSONResponse(build_test_data_payload())


# --------------------------------------------------
# Prediction Endpoint (Protected)
# --------------------------------------------------
@app.post("/predict")
def predict(
    request: PredictionRequest,
    auth: bool = Depends(verify_password)
):
    return run_prediction(request.features)


@app.post("/predict-ui")
def predict_ui(request: PredictionRequest):
    return run_prediction(request.features)


# --------------------------------------------------
# Run Server
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
