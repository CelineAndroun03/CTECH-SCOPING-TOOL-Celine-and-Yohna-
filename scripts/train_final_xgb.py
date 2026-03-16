import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


# ============================================================
# FINAL LOCKED SETTINGS
# ============================================================

TARGET_COL = "Eng. AH"
HUB_COL = "CCN_Data Hub"
TOPK_HUB = 10
ARTIFACT_DIR = "artifacts"
DATA_FILE = r"Code\1. Exploratory Data Analysis\Final_data_Ctech.xlsx" 
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

DROP_COLS = [
    "22 (Derek Understand when used)",
    "UL", "cUL", "CCC", "KC", "PSE", "NOM",
    "CCN_Top10_AAAL", "CCN_Top10_AAAU", "CCN_Top10_NWIN",
    "CCN_Top10_OTHER", "CCN_Top10_QQGQ2",
    "total_schemmes",
    "uncertified_pwr_supply", "preliminary_review", "7 (950 TO 368)",
    "IT Informational Test Report", "touch_current_test", "all_mount_test",
    "enclosure_push_test", "energy_hazard_test", "water_spray_test",
    "thin_material_test", "stability_test", "strain_relief_test", "dust_test",
    "varistor_overload_test", "102 (LFC)",
    "103 (L R Over DC Mot in Sec Cir)", "moment_test",
    "Test- Bridging Resistor Test", "Test- Tensile Strength",
    "Test- Secondary Working Voltage", "outdoor_use", "PoE_source", "PoE_load",
    "multilayer_boards", "unknown_PWR_supply",
    "\n33 (Rename)",
    "alter_transformer", "alter_capacitor", "alter_layout", "alter_insulation",
    "alter_optical_isolator", "alter_inductor", "alter_resistor", "alter_movs",
    "alter_LPS", "alter_connector", "alter_app_inlet", "alter_relay",
    "alter_cord", "alter_current_ic", "alter_strain_relief", "alter_ptc",
    "alter_display", "alter_interconnect", "alter_light",
    "Investigation_type_4 - DC Distribution Panels",
    "type_of_investigation_5 - Administrative CB review",
    "9 (Talk to Derek)", "4 (Annex Y)",
    "11 (Ask Derek if  IEC/EN 62368-3  covers this)",
    "34 (may be duplicate - check)",
    "30 (Talk to Derek)", " 20 (Iteration of Tests ASK Derek)",
    "29 (Talk to Derek)", "27 (Change Enc Func)",
    "_60950_1_2ed_A2", "capacitance_discharge_test",
    "total_CB_count_Binary", "total_test_count_Binary", "standard_count_Binary"
]

# IMPORTANT:
# These must match your actual RFECV-selected feature names exactly.
SELECTED_FEATURES = [
    "standard_count",
    "1 (60950-1)",
    "total_CB_count",
    "total_test_count",
    "Region_AMERICAS",
    "Region_ASIA",
    "Investigation_type_2 - Class III",
    "Investigation_type_3 - Power Supply",
    "type_of_investigation_1 - Full Investigation",
    "type_of_investigation_2 - Full Investigation + Alternate Construction",
    "type_of_investigation_3 - Alternate Construction",
    "type_of_investigation_4 - Administrative No Test anticipated (revisions requiring Engineering Review)",
    "CCN_Top10_AALL",
    "CCN_Top10_AZOT",
    "CCN_Top10_AZOT2",
    "CCN_Top10_QQJQ",
    "CCN_Top10_QQJQ2"
]

NUMERIC_COLS = [
    TARGET_COL,
    "total_CB_count",
    "total_test_count",
    "standard_count"
]


# ============================================================
# UTILITIES
# ============================================================

def ensure_artifact_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj, filepath: str) -> None:
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=2)


def prepare_target(df: pd.DataFrame) -> pd.Series:
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    return y


def fit_preprocessor(df_train: pd.DataFrame) -> dict:
    """
    Learn anything that must be frozen from training data only.
    """
    temp = df_train.copy()

    top_hubs = []
    if HUB_COL in temp.columns:
        temp[HUB_COL] = temp[HUB_COL].fillna("Missing").astype(str).str.strip()
        hub_counts = temp[HUB_COL].value_counts()
        hub_counts = hub_counts.drop(labels=["0"], errors="ignore")
        top_hubs = hub_counts.head(TOPK_HUB).index.tolist()

    config = {
        "target_col": TARGET_COL,
        "hub_col": HUB_COL,
        "topk_hub": TOPK_HUB,
        "top_hubs": top_hubs,
        "drop_cols": DROP_COLS,
        "selected_features": SELECTED_FEATURES,
        "numeric_cols": NUMERIC_COLS
    }
    return config


def preprocess_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply frozen preprocessing to any dataframe.
    """
    temp = df.copy()

    # Drop known unused columns
    temp = temp.drop(columns=config["drop_cols"], errors="ignore")

    # Clean categoricals
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

    # Numeric conversion
    for col in config["numeric_cols"]:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce")

    # Hub grouping
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

    # One-hot encoding
    if "Region" in temp.columns:
        region_ohe = pd.get_dummies(temp["Region"], prefix="Region").astype(int)
        temp = pd.concat([temp, region_ohe], axis=1)

    if "Investigation_type" in temp.columns:
        inv_ohe = pd.get_dummies(temp["Investigation_type"], prefix="Investigation_type").astype(int)
        temp = pd.concat([temp, inv_ohe], axis=1)

    if "type_of_investigation" in temp.columns:
        toi_ohe = pd.get_dummies(temp["type_of_investigation"], prefix="type_of_investigation").astype(int)
        temp = pd.concat([temp, toi_ohe], axis=1)

    ccn_ohe = pd.get_dummies(temp["CCN_Hub_top10"], prefix="CCN_Top10").astype(int)
    temp = pd.concat([temp, ccn_ohe], axis=1)

    # Drop original categoricals after encoding
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

    # Keep predictor columns only
    X = temp.drop(columns=[config["target_col"]], errors="ignore")

    # Only numeric columns
    X = X.select_dtypes(include=[np.number]).copy()

    # Fill remaining NaNs
    X = X.fillna(0)

    return X


def validate_selected_features(X: pd.DataFrame, selected_features: list) -> None:
    """
    Fail early with a clear message if selected feature names do not exist.
    """
    missing = [col for col in selected_features if col not in X.columns]
    if missing:
        raise ValueError(
            "These selected feature names were not found after preprocessing:\n"
            + "\n".join(missing)
            + "\n\nCheck the exact spellings in SELECTED_FEATURES."
        )


def align_selected_features(X: pd.DataFrame, selected_features: list) -> pd.DataFrame:
    """
    Force exact final feature order.
    """
    return X.reindex(columns=selected_features, fill_value=0)


def evaluate_model(model, X: pd.DataFrame, y_log: pd.Series) -> dict:
    pred_log = model.predict(X)

    r2_log = r2_score(y_log, pred_log)

    y_hours = np.exp(y_log)
    pred_hours = np.exp(pred_log)

    mae_hours = mean_absolute_error(y_hours, pred_hours)
    rmse_hours = np.sqrt(mean_squared_error(y_hours, pred_hours))

    return {
        "R2_log": float(r2_log),
        "MAE_hours": float(mae_hours),
        "RMSE_hours": float(rmse_hours)
    }


def predict_from_dataframe(model, config: dict, df: pd.DataFrame) -> pd.DataFrame:
    X = preprocess_features(df, config)
    X = align_selected_features(X, config["selected_features"])
    pred_log = model.predict(X)
    pred_hours = np.exp(pred_log)

    output = df.copy()
    output["predicted_Eng_AH"] = pred_hours
    return output


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("FINAL XGBOOST TRAINING PIPELINE")
    print("=" * 70)

    # 1) Load data
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Could not find '{DATA_FILE}'. Put the Excel file in the same folder as this script."
        )

    data = pd.read_excel(DATA_FILE)
    print("Raw shape:", data.shape)

    # 2) Drop duplicates
    data = data.drop_duplicates().copy()
    print("After dropping duplicate rows:", data.shape)

    # 3) Clean target
    data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
    data = data.dropna(subset=[TARGET_COL]).copy()
    data = data[data[TARGET_COL] > 0].copy()
    print("After target cleaning:", data.shape)

    if data.empty:
        raise ValueError("No rows left after cleaning target column.")

    # 4) Split raw data first
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

    # 5) Fit preprocessing config on train only
    config = fit_preprocessor(train_df)

    # 6) Preprocess
    X_train = preprocess_features(train_df, config)
    X_val = preprocess_features(val_df, config)
    X_test = preprocess_features(test_df, config)

    # 7) Validate selected feature names before aligning
    validate_selected_features(X_train, config["selected_features"])

    # 8) Align to final selected features
    X_train = align_selected_features(X_train, config["selected_features"])
    X_val = align_selected_features(X_val, config["selected_features"])
    X_test = align_selected_features(X_test, config["selected_features"])

    # 9) Prepare target
    y_train_raw = prepare_target(train_df)
    y_val_raw = prepare_target(val_df)
    y_test_raw = prepare_target(test_df)

    train_mask = y_train_raw.notna() & (y_train_raw > 0)
    val_mask = y_val_raw.notna() & (y_val_raw > 0)
    test_mask = y_test_raw.notna() & (y_test_raw > 0)

    X_train = X_train.loc[train_mask].copy()
    X_val = X_val.loc[val_mask].copy()
    X_test = X_test.loc[test_mask].copy()

    y_train_raw = y_train_raw.loc[train_mask].copy()
    y_val_raw = y_val_raw.loc[val_mask].copy()
    y_test_raw = y_test_raw.loc[test_mask].copy()

    y_train = np.log(y_train_raw)
    y_val = np.log(y_val_raw)
    y_test = np.log(y_test_raw)

    print("\nFinal matrices:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    # 10) Train final model
    model = XGBRegressor(**FINAL_XGB_PARAMS)
    model.fit(X_train, y_train)

    # 11) Evaluate
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)

    print("\n" + "=" * 70)
    print("FINAL MODEL RESULTS")
    print("=" * 70)
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

    # 12) Save artifacts
    ensure_artifact_dir(ARTIFACT_DIR)

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
        "selected_features": config["selected_features"]
    }
    save_json(metrics, metrics_path)

    print("\nArtifacts saved:")
    print("Model:", model_path)
    print("Config:", config_path)
    print("Params:", params_path)
    print("Metrics:", metrics_path)

    # 13) Demo predictions
    demo_rows = test_df.head(3).copy()
    demo_preds = predict_from_dataframe(model, config, demo_rows)

    print("\nSample predictions:")
    cols_to_show = [col for col in [TARGET_COL, "predicted_Eng_AH"] if col in demo_preds.columns]
    print(demo_preds[cols_to_show].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
