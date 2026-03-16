import json
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ARTIFACT_DIR = "artifacts"
MODEL_PATH = f"{ARTIFACT_DIR}/xgb_model.pkl"
CONFIG_PATH = f"{ARTIFACT_DIR}/preprocessing_config.json"


# ============================================================
# LOAD ARTIFACTS
# ============================================================

def load_artifacts():
    model = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    return model, config


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    temp = df.copy()

    # Drop columns not needed
    temp = temp.drop(columns=config["drop_cols"], errors="ignore")

    # Clean categorical columns
    if "Region" in temp.columns:
        temp["Region"] = temp["Region"].fillna("Missing").astype(str).str.strip()

    if "Investigation_type" in temp.columns:
        temp["Investigation_type"] = (
            temp["Investigation_type"].fillna("Missing").astype(str).str.strip()
        )

    if "type_of_investigation" in temp.columns:
        temp["type_of_investigation"] = (
            temp["type_of_investigation"].fillna("Missing").astype(str).str.strip()
        )

    # Convert numeric columns
    for col in config["numeric_cols"]:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce")

    # Group CCN hubs
    if config["hub_col"] in temp.columns:
        temp[config["hub_col"]] = (
            temp[config["hub_col"]].fillna("Missing").astype(str).str.strip()
        )

        temp["CCN_Hub_top10"] = np.where(
            temp[config["hub_col"]].isin(config["top_hubs"]),
            temp[config["hub_col"]],
            "OTHER"
        )
    else:
        temp["CCN_Hub_top10"] = "OTHER"

    # One-hot encode Region
    if "Region" in temp.columns:
        region_ohe = pd.get_dummies(temp["Region"], prefix="Region").astype(int)
        temp = pd.concat([temp, region_ohe], axis=1)

    # One-hot encode Investigation_type
    if "Investigation_type" in temp.columns:
        inv_ohe = pd.get_dummies(
            temp["Investigation_type"],
            prefix="Investigation_type"
        ).astype(int)
        temp = pd.concat([temp, inv_ohe], axis=1)

    # One-hot encode type_of_investigation
    if "type_of_investigation" in temp.columns:
        toi_ohe = pd.get_dummies(
            temp["type_of_investigation"],
            prefix="type_of_investigation"
        ).astype(int)
        temp = pd.concat([temp, toi_ohe], axis=1)

    # One-hot encode CCN grouped hubs
    ccn_ohe = pd.get_dummies(temp["CCN_Hub_top10"], prefix="CCN_Top10").astype(int)
    temp = pd.concat([temp, ccn_ohe], axis=1)

    # Drop original categorical columns after encoding
    temp = temp.drop(
        columns=[
            "Region",
            "Investigation_type",
            "type_of_investigation",
            config["hub_col"],
            "CCN_Hub_top10"
        ],
        errors="ignore"
    )

    # Drop target if present
    X = temp.drop(columns=[config["target_col"]], errors="ignore")

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number]).copy()

    # Fill missing numeric values
    X = X.fillna(0)

    # Force exact feature order used during training
    X = X.reindex(columns=config["selected_features"], fill_value=0)

    return X


# ============================================================
# PREDICTION
# ============================================================

def predict_hours(df: pd.DataFrame) -> pd.DataFrame:
    model, config = load_artifacts()

    X = preprocess_features(df, config)

    pred_log = model.predict(X)
    pred_hours = np.exp(pred_log)

    output = df.copy()
    output["predicted_Eng_AH"] = pred_hours

    return output


# ============================================================
# TEST RUN
# ============================================================

if __name__ == "__main__":
    sample_input = pd.DataFrame([
        {
            "standard_count": 1,
            "total_CB_count": 2,
            "total_test_count": 3,
            "Region": "AMERICAS",
            "Investigation_type": "2 - Class III",
            "type_of_investigation": "1 - Full Investigation",
            "CCN_Data Hub": "AALL",
            "1 (60950-1)": 1
        },
        {
            "standard_count": 2,
            "total_CB_count": 1,
            "total_test_count": 4,
            "Region": "ASIA",
            "Investigation_type": "3 - Power Supply",
            "type_of_investigation": "3 - Alternate Construction",
            "CCN_Data Hub": "AZOT",
            "1 (60950-1)": 0
        }
    ])

    predictions = predict_hours(sample_input)

    print("\nPredictions:")
    print(predictions.to_string(index=False))
