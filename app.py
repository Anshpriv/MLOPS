import pickle
import logging
import uvicorn
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional

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
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "templates" / "index.html"
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

# --------------------------------------------------
# Startup Event – Load Model
# --------------------------------------------------
@app.on_event("startup")
def load_model():
    global model
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)

        if not hasattr(model, "predict"):
            raise AttributeError("Loaded object does not have a predict() method")

        logging.info("Model loaded successfully.")

    except FileNotFoundError:
        logging.error("model.pkl not found. Please place the model file in the root directory.")
        model = None

    except pickle.UnpicklingError:
        logging.error("Error while unpickling the model file. The file may be corrupted.")
        model = None

    except AttributeError as e:
        logging.error(f"Invalid model object: {e}")
        model = None

    except Exception as e:
        logging.error(f"Unexpected error while loading model: {e}")
        model = None


# --------------------------------------------------
# Request Schema
# --------------------------------------------------
class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_items=4,
        max_items=4,
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
