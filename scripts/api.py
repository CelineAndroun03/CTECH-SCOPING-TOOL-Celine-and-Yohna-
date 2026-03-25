from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
from scripts.predict import predict_hours

app = FastAPI(title="CTECH Effort Estimation API - Predict Engineering Actual Hours")

# ============================================================
# REQUEST MODEL
# ============================================================
class PredictionInput(BaseModel):
    standard_count: float
    total_CB_count: float
    total_test_count: float
    Region: str
    Investigation_type: str
    type_of_investigation: str
    CCN_Data_Hub: str
    _60950_1: int  # maps to "1 (60950-1)"

# ============================================================
# HELPER FUNCTION
# ============================================================

def transform_input(data):
    """
    Convert API input into dataframe matching training format
    """
    df = pd.DataFrame(data)

    # Rename fields to match model
    df = df.rename(columns={
        "CCN_Data_Hub": "CCN_Data Hub",
        "_60950_1": "1 (60950-1)"
    })

    return df

# # ============================================================
# # HEALTH CHECK
# # ============================================================

# @app.get("/")
# def health():
#     return {"status": "API is running"}

# ============================================================
# BATCH PREDICTION
# ============================================================

@app.post("/Predict Eng Hours", tags=["Predictions"])
def predict_batch(input_data: List[PredictionInput]):
    try:
        df = transform_input([item.dict() for item in input_data])
        result = predict_hours(df)

        return {
            "predictions": [float(x) for x in result["predicted_Eng_AH"].tolist()]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
