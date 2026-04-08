import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ============================================================
# PATHS / SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cleaned file created by cleaning_v1.py
DATA_FILE = os.path.join(BASE_DIR, "artifacts", "lab_cleaned_final.xlsx")

# Save trained model artifacts here
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

TARGET_COL = "Lab. AH"
LOG_TARGET_COL = "Lab_AH_log"
RANDOM_STATE = 42

FINAL_XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# ============================================================
# FINAL SELECTED FEATURES FOR LAB (4-INPUT TEST)
# ============================================================

SELECTED_FEATURES = [
    "standard_count",
    "total_CB_count",
    "total_test_count",
    "1 (60950-1)"
]

NUMERIC_COLS = [
    "standard_count",
    "total_CB_count",
    "total_test_count",
    "1 (60950-1)"
]

# ============================================================
# UTILITIES
# ============================================================

def save_json(obj, filepath: str) -> None:
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=4)

def validate_required_columns(df: pd.DataFrame, required_cols: list) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "These required columns are missing from the training dataset:\n"
            + "\n".join(missing)
        )

def prepare_target(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df[LOG_TARGET_COL], errors="coerce")

def fit_preprocessor() -> dict:
    config = {
        "target_col": TARGET_COL,
        "log_target_col": LOG_TARGET_COL,
        "selected_features": SELECTED_FEATURES,
        "numeric_cols": NUMERIC_COLS,
        "total_cb_binary": True,
        "standard_count_int": True,
        "feature_version": "LAB_4_INPUT_TEST"
    }
    return config

def preprocess_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    temp = df.copy()

    # Convert numeric columns
    for col in config["numeric_cols"]:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce")

    # Match cleaning / prediction logic exactly
    if "total_CB_count" in temp.columns and config.get("total_cb_binary", False):
        temp["total_CB_count"] = (temp["total_CB_count"].fillna(0) > 0).astype(int)

    if "standard_count" in temp.columns and config.get("standard_count_int", False):
        temp["standard_count"] = temp["standard_count"].fillna(0).astype(int)

    if "total_test_count" in temp.columns:
        temp["total_test_count"] = temp["total_test_count"].fillna(0)

    if "1 (60950-1)" in temp.columns:
        temp["1 (60950-1)"] = temp["1 (60950-1)"].fillna(0).astype(int)

    # Force exact feature order
    X = temp.reindex(columns=config["selected_features"], fill_value=0).copy()
    X = X.fillna(0)

    return X

def evaluate_model(model, X: pd.DataFrame, y_log: pd.Series) -> dict:
    pred_log = model.predict(X)

    r2_log = r2_score(y_log, pred_log)

    y_hours = np.expm1(y_log)
    pred_hours = np.expm1(pred_log)

    mae_hours = mean_absolute_error(y_hours, pred_hours)
    rmse_hours = np.sqrt(mean_squared_error(y_hours, pred_hours))

    return {
        "R2_log": float(r2_log),
        "MAE_hours": float(mae_hours),
        "RMSE_hours": float(rmse_hours)
    }

def predict_from_dataframe(model, config: dict, df: pd.DataFrame) -> pd.DataFrame:
    X = preprocess_features(df, config)
    pred_log = model.predict(X)
    pred_hours = np.expm1(pred_log)

    output = df.copy()
    output["predicted_Lab_AH"] = pred_hours
    return output

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("FINAL LAB XGBOOST TRAINING PIPELINE (4-INPUT TEST)")
    print("=" * 80)

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DATA_FILE: {os.path.abspath(DATA_FILE)}")
    print(f"ARTIFACT_DIR: {os.path.abspath(ARTIFACT_DIR)}")

    # --------------------------------------------------------
    # 1) Load cleaned data
    # --------------------------------------------------------
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Could not find training file at:\n{os.path.abspath(DATA_FILE)}"
        )

    data = pd.read_excel(DATA_FILE)
    print("\nRaw shape:", data.shape)

    # --------------------------------------------------------
    # 2) Drop duplicate rows
    # --------------------------------------------------------
    data = data.drop_duplicates().copy()
    print("After dropping duplicate rows:", data.shape)

    # --------------------------------------------------------
    # 3) Validate required columns
    # --------------------------------------------------------
    required_cols = [LOG_TARGET_COL] + SELECTED_FEATURES
    validate_required_columns(data, required_cols)

    # --------------------------------------------------------
    # 4) Clean target
    # --------------------------------------------------------
    data[LOG_TARGET_COL] = pd.to_numeric(data[LOG_TARGET_COL], errors="coerce")
    data = data.dropna(subset=[LOG_TARGET_COL]).copy()

    print("After target cleaning:", data.shape)

    if data.empty:
        raise ValueError("No rows left after cleaning the target column.")

    # --------------------------------------------------------
    # 5) Train / Validation / Test split
    # --------------------------------------------------------
    train_df, temp_df = train_test_split(
        data,
        test_size=0.30,
        random_state=RANDOM_STATE
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    print("Train raw shape:", train_df.shape)
    print("Val raw shape:", val_df.shape)
    print("Test raw shape:", test_df.shape)

    # --------------------------------------------------------
    # 6) Fit preprocessing config
    # --------------------------------------------------------
    config = fit_preprocessor()

    # --------------------------------------------------------
    # 7) Preprocess features
    # --------------------------------------------------------
    X_train = preprocess_features(train_df, config)
    X_val = preprocess_features(val_df, config)
    X_test = preprocess_features(test_df, config)

    # --------------------------------------------------------
    # 8) Prepare targets
    # --------------------------------------------------------
    y_train = prepare_target(train_df)
    y_val = prepare_target(val_df)
    y_test = prepare_target(test_df)

    train_mask = y_train.notna()
    val_mask = y_val.notna()
    test_mask = y_test.notna()

    X_train = X_train.loc[train_mask].copy()
    X_val = X_val.loc[val_mask].copy()
    X_test = X_test.loc[test_mask].copy()

    y_train = y_train.loc[train_mask].copy()
    y_val = y_val.loc[val_mask].copy()
    y_test = y_test.loc[test_mask].copy()

    print("\nFinal matrices:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    print("\nTraining features used:")
    print(X_train.columns.tolist())

    # --------------------------------------------------------
    # 9) Train model
    # --------------------------------------------------------
    model = XGBRegressor(**FINAL_XGB_PARAMS)
    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # 10) Evaluate
    # --------------------------------------------------------
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)

    print("\n" + "=" * 80)
    print("FINAL LAB MODEL RESULTS")
    print("=" * 80)
    print(
        f"Train -> R2={train_metrics['R2_log']:.3f}, "
        f"MAE={train_metrics['MAE_hours']:.2f}h, "
        f"RMSE={train_metrics['RMSE_hours']:.2f}h"
    )
    print(
        f"Val   -> R2={val_metrics['R2_log']:.3f}, "
        f"MAE={val_metrics['MAE_hours']:.2f}h, "
        f"RMSE={val_metrics['RMSE_hours']:.2f}h"
    )
    print(
        f"Test  -> R2={test_metrics['R2_log']:.3f}, "
        f"MAE={test_metrics['MAE_hours']:.2f}h, "
        f"RMSE={test_metrics['RMSE_hours']:.2f}h"
    )

    # --------------------------------------------------------
    # 11) Save artifacts
    # --------------------------------------------------------
    model_path = os.path.join(ARTIFACT_DIR, "xgb_model.pkl")
    config_path = os.path.join(ARTIFACT_DIR, "preprocessing_config.json")
    params_path = os.path.join(ARTIFACT_DIR, "best_xgb_params.json")
    metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")

    joblib.dump(model, model_path)
    save_json(config, config_path)
    save_json(FINAL_XGB_PARAMS, params_path)

    metrics = {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "n_rows_total_after_target_cleaning": int(len(data)),
        "n_rows_train": int(len(X_train)),
        "n_rows_val": int(len(X_val)),
        "n_rows_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "selected_features": config["selected_features"],
        "feature_version": config["feature_version"]
    }
    save_json(metrics, metrics_path)

    print("\nArtifacts saved:")
    print("Model :", model_path)
    print("Config:", config_path)
    print("Params:", params_path)
    print("Metrics:", metrics_path)

    # --------------------------------------------------------
    # 12) Demo predictions
    # --------------------------------------------------------
    demo_rows = test_df.head(3).copy()
    demo_preds = predict_from_dataframe(model, config, demo_rows)

    print("\nSample predictions:")
    cols_to_show = [col for col in [TARGET_COL, "predicted_Lab_AH"] if col in demo_preds.columns]
    print(demo_preds[cols_to_show].to_string(index=False))

    print("\nDone.")

if __name__ == "__main__":
    main()
