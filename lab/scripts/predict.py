import json
import os
import joblib
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACT_DIR, "xgb_model.pkl")
CONFIG_PATH = os.path.join(ARTIFACT_DIR, "preprocessing_config.json")


# ============================================================
# LOAD ARTIFACTS
# ============================================================

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    model = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    return model, config


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    temp = df.copy()

    # Convert numeric columns
    for col in config["numeric_cols"]:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce")

    # Match training logic
    if "total_CB_count" in temp.columns and config.get("total_cb_binary", False):
        temp["total_CB_count"] = (temp["total_CB_count"].fillna(0) > 0).astype(int)

    if "standard_count" in temp.columns and config.get("standard_count_int", False):
        temp["standard_count"] = temp["standard_count"].fillna(0).astype(int)

    if "total_test_count" in temp.columns:
        temp["total_test_count"] = temp["total_test_count"].fillna(0)

    if "1 (60950-1)" in temp.columns:
        temp["1 (60950-1)"] = temp["1 (60950-1)"].fillna(0)

    if "Lab. SH" in temp.columns and config.get("lab_sh_fillna_zero", False):
        temp["Lab. SH"] = temp["Lab. SH"].fillna(0)

    # Force exact feature order used in training
    X = temp.reindex(columns=config["selected_features"], fill_value=0).copy()
    X = X.fillna(0)

    return X


# ============================================================
# PREDICTION
# ============================================================

def predict_hours(df: pd.DataFrame) -> pd.DataFrame:
    model, config = load_artifacts()

    X = preprocess_features(df, config)

    pred_log = model.predict(X)
    pred_hours = np.expm1(pred_log)

    output = df.copy()
    output["predicted_Lab_AH"] = pred_hours

    return output


# ============================================================
# TEST RUN
# ============================================================

if __name__ == "__main__":
    # Example local test
    sample = pd.DataFrame([
        {
            "1 (60950-1)": 1,
            "Lab. SH": 3,
            "standard_count": 2,
            "total_CB_count": 1,
            "total_test_count": 4
        }
    ])

    result = predict_hours(sample)
    print(result.to_string(index=False))
