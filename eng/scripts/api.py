from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd

from eng.scripts.predict import predict_hours as predict_eng_hours
from lab.scripts.predict import predict_hours as predict_lab_hours

app = FastAPI(title="CTECH Effort Estimation API")


# ============================================================
# ENGINEERING REQUEST MODEL
# ============================================================

class EngPredictionInput(BaseModel):
    standard_count: float
    total_CB_count: float
    total_test_count: float
    Region: str
    Investigation_type: str
    type_of_investigation: str
    CCN_Data_Hub: str
    stan_60950_1: int


# ============================================================
# LAB REQUEST MODEL
# ============================================================

class LabPredictionInput(BaseModel):
    standard_count: float
    total_CB_count: float
    total_test_count: float
    # Lab_SH: float
    stan_60950_1: int


# ============================================================
# HELPERS
# ============================================================

def transform_eng_input(data):
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "CCN_Data_Hub": "CCN_Data Hub",
        "stan_60950_1": "1 (60950-1)"
    })
    return df


def transform_lab_input(data):
    df = pd.DataFrame(data)
    df = df.rename(columns={
        # "Lab_SH": "Lab. SH",
        "stan_60950_1": "1 (60950-1)"
    })
    return df


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def health():
    return {"status": "API is running"}


# ============================================================
# ENGINEERING BATCH PREDICTION
# ============================================================

@app.post("/predict-eng-hours", tags=["Engineering"])
def predict_eng_batch(input_data: List[EngPredictionInput]):
    try:
        df = transform_eng_input([item.model_dump() for item in input_data])
        result = predict_eng_hours(df)

        return {
            "predictions": [float(x) for x in result["predicted_Eng_AH"].tolist()]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# LAB BATCH PREDICTION
# ============================================================

@app.post("/predict-lab-hours", tags=["Lab"])
def predict_lab_batch(input_data: List[LabPredictionInput]):
    try:
        df = transform_lab_input([item.model_dump() for item in input_data])
        result = predict_lab_hours(df)

        return {
            "predictions": [float(x) for x in result["predicted_Lab_AH"].tolist()]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
