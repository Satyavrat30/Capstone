from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pkl"
STATIC_DIR = BASE_DIR / "static"
DATA_PATH = BASE_DIR / "housing.csv"


FEATURE_NAMES: List[str] = [
    "crim",
    "zn",
    "indus",
    "chas",
    "nox",
    "rm",
    "age",
    "dis",
    "rad",
    "tax",
    "ptratio",
    "b",
    "lstat",
]


class HouseInput(BaseModel):
    crim: float = Field(..., ge=0)
    zn: float = Field(..., ge=0)
    indus: float = Field(..., ge=0)
    chas: float = Field(..., ge=0, le=1)
    nox: float = Field(..., ge=0)
    rm: float = Field(..., ge=0)
    age: float = Field(..., ge=0)
    dis: float = Field(..., ge=0)
    rad: float = Field(..., ge=0)
    tax: float = Field(..., ge=0)
    ptratio: float = Field(..., ge=0)
    b: float = Field(..., ge=0)
    lstat: float = Field(..., ge=0)


def train_and_save_model() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    lower_map = {c.lower().strip(): c for c in df.columns}

    # Case 1: dataset already has the expected Boston features.
    if set(FEATURE_NAMES).issubset(df.columns):
        x = df[FEATURE_NAMES].to_numpy(dtype=float)
    # Case 2: dataset has same names with different casing/spacing.
    elif set(FEATURE_NAMES).issubset(set(lower_map.keys())):
        cols = [lower_map[name] for name in FEATURE_NAMES]
        x = df[cols].to_numpy(dtype=float)
    else:
        raise ValueError("housing.csv does not contain required feature columns.")

    if "medv" in df.columns:
        y = df["medv"].to_numpy(dtype=float) * 1000.0
    elif "MEDV" in df.columns:
        y = df["MEDV"].to_numpy(dtype=float) * 1000.0
    else:
        raise ValueError("housing.csv does not contain target column: medv")

    model = LinearRegression()
    model.fit(x, y)
    joblib.dump(model, MODEL_PATH, compress=3)


def load_or_create_model():
    max_size_bytes = 50 * 1024 * 1024
    need_retrain = (
        (not MODEL_PATH.exists())
        or (MODEL_PATH.stat().st_size > max_size_bytes)
        or (DATA_PATH.exists() and MODEL_PATH.stat().st_mtime < DATA_PATH.stat().st_mtime)
    )
    if need_retrain:
        train_and_save_model()
    return joblib.load(MODEL_PATH)


app = FastAPI(title="House Predictor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

model = load_or_create_model()


@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/predict")
def predict_price(payload: HouseInput):
    row = np.array(
        [
            [
                payload.crim,
                payload.zn,
                payload.indus,
                payload.chas,
                payload.nox,
                payload.rm,
                payload.age,
                payload.dis,
                payload.rad,
                payload.tax,
                payload.ptratio,
                payload.b,
                payload.lstat,
            ]
        ],
        dtype=float,
    )
    prediction = float(model.predict(row)[0])
    return {"predicted_price": round(prediction, 2), "currency": "USD"}
