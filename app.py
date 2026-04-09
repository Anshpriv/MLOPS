import pickle
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, Header, Depends
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
    title="Model Prediction API",
    description="API for ML model prediction with simple HTTP auth",
    version="1.1.0"
)

# --------------------------------------------------
# Global Model Variable
# --------------------------------------------------
model = None

# --------------------------------------------------
# Simple Header Authentication Dependency
# --------------------------------------------------
def verify_password(password: Optional[str] = Header(None)):
    if password != "Ansh":
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing password header"
        )
    return True

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
        min_items=1,
        description="List of numerical features for prediction"
    )


# --------------------------------------------------
# Prediction Endpoint (Protected)
# --------------------------------------------------
@app.post("/predict")
def predict(
    request: PredictionRequest,
    auth: bool = Depends(verify_password)  # 🔐 Auth applied here
):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded. Please contact the administrator."
        )

    if not request.features:
        raise HTTPException(
            status_code=422,
            detail="Feature list cannot be empty."
        )

    try:
        prediction = model.predict([request.features])

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
# Run Server
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )