from contextlib import asynccontextmanager
from pathlib import Path
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from adult_income.model import predict_records

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("ADULT_MODEL_PATH", str(ROOT / "artifacts/pipeline.joblib")))

@asynccontextmanager
async def lifespan(app):
    app.state.pipeline = joblib.load(MODEL_PATH) if MODEL_PATH.is_file() else None
    yield
    app.state.pipeline = None

app = FastAPI(title="Adult Income demonstration API", lifespan=lifespan)

class InputFeatures(BaseModel):
    age: int = Field(ge=0, le=120)
    workclass: str
    fnlwgt: int = Field(ge=0)
    education: str
    education_num: int = Field(ge=0)
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(ge=0)
    capital_loss: int = Field(ge=0)
    hours_per_week: int = Field(ge=0, le=168)
    country: str

@app.get("/")
def home():
    return {"message": "Adult Income demonstration API", "model_ready": app.state.pipeline is not None}

@app.post("/predict")
def predict(features: InputFeatures):
    if app.state.pipeline is None:
        raise HTTPException(status_code=503, detail="Train the model with train.py before requesting predictions.")
    return predict_records(app.state.pipeline, pd.DataFrame([features.model_dump()]))[0]
