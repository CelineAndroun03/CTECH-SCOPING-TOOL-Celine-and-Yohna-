import json
import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "xgb_model.pkl")
CONFIG_PATH = os.path.join(ARTIFACT_DIR, "preprocessing_config.json")
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
    # Load full dataset
    data_path = os.path.join(BASE_DIR, "Code", "1. Exploratory Data Analysis", "Final_data_Ctech.xlsx")
    df = pd.read_excel(data_path)
    print("\n===== Raw Dataset Info =====")
    print(f"Dataset shape: {df.shape}")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")


    print("Loaded data shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Keep only rows where actual Eng. AH exists
    df = df.dropna(subset=["Eng. AH"]).copy()

    # Run predictions
    predictions = predict_hours(df)

    print("\nPredictions sample:")
    print(predictions[["predicted_Eng_AH"]].head().to_string(index=False))

    # --------------------------------------------------
    # Accuracy evaluation
    # --------------------------------------------------
    y_true = pd.to_numeric(df["Eng. AH"], errors="coerce")
    y_pred = pd.to_numeric(predictions["predicted_Eng_AH"], errors="coerce")

    error = np.abs(y_true - y_pred)

    print("\n===== Accuracy Distribution =====")
    print(f"Total rows scored: {len(error)}")
    print(f"MAE: {np.mean(error):.2f} hours")

    print(f"% within 1 hour: {np.mean(error <= 1) * 100:.1f}%")
    print(f"% within 2 hours: {np.mean(error <= 2) * 100:.1f}%")
    print(f"% within 3 hours: {np.mean(error <= 3) * 100:.1f}%")

    print(f"90% of predictions within: {np.percentile(error, 90):.2f} hours")
    print(f"95% of predictions within: {np.percentile(error, 95):.2f} hours")

