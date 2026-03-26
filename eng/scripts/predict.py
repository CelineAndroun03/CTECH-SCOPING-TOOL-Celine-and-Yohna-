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
    data_path = os.path.join(BASE_DIR, "scripts", "1. Exploratory Data Analysis", "Final_data_Ctech.xlsx")
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

# Drop bad rows if any
mask = y_true.notna() & y_pred.notna()
y_true = y_true[mask]
y_pred = y_pred[mask]

# Absolute error in hours
error = np.abs(y_true - y_pred)

# Relative error as % of actual hours
# Avoid divide-by-zero if any actual hours are 0
pct_error = np.where(y_true != 0, (error / y_true) * 100, np.nan)

print("\n===== Accuracy Distribution =====")
print(f"Total rows scored: {len(error)}")
print(f"MAE: {np.mean(error):.2f} hours")
print(f"Median absolute error: {np.median(error):.2f} hours")

print(f"% within 1 hour: {np.mean(error <= 1) * 100:.1f}%")
print(f"% within 2 hours: {np.mean(error <= 2) * 100:.1f}%")
print(f"% within 3 hours: {np.mean(error <= 3) * 100:.1f}%")

print(f"90% of predictions within: {np.percentile(error, 90):.2f} hours")
print(f"95% of predictions within: {np.percentile(error, 95):.2f} hours")

print("\n===== Relative Error (Percentage of Actual Hours) =====")
print(f"Mean % error: {np.nanmean(pct_error):.2f}%")
print(f"Median % error: {np.nanmedian(pct_error):.2f}%")
print(f"% within 10%: {np.nanmean(pct_error <= 10) * 100:.1f}%")
print(f"% within 20%: {np.nanmean(pct_error <= 20) * 100:.1f}%")
print(f"% within 30%: {np.nanmean(pct_error <= 30) * 100:.1f}%")
print(f"90% of predictions within: {np.nanpercentile(pct_error, 90):.2f}%")
print(f"95% of predictions within: {np.nanpercentile(pct_error, 95):.2f}%")

# --------------------------------------------------
# Business-friendly breakdown by hour-error bucket
# This answers: "1, 2, 3 hours off... on projects of what size?"
# --------------------------------------------------

print("100% of the dataset results")
summary_df = pd.DataFrame({
    "actual_hours": y_true.values,
    "predicted_hours": y_pred.values,
    "abs_error_hours": error.values,
    "pct_error": pct_error
})

bucket_definitions = [
    ("≤ 1 hour off", summary_df["abs_error_hours"] <= 1),
    ("1–2 hours off", (summary_df["abs_error_hours"] > 1) & (summary_df["abs_error_hours"] <= 2)),
    ("2–3 hours off", (summary_df["abs_error_hours"] > 2) & (summary_df["abs_error_hours"] <= 3)),
    ("3–7 hours off", (summary_df["abs_error_hours"] > 3) & (summary_df["abs_error_hours"] <= 7)),
    ("> 7 hours off", summary_df["abs_error_hours"] > 7),
]

print("\n===== Error Buckets with Project Size Context =====")
for label, condition in bucket_definitions:
    group = summary_df[condition]
    if len(group) == 0:
        print(f"{label}: 0 rows")
        continue

    print(f"\n{label}")
    print(f"Count: {len(group)} ({len(group) / len(summary_df) * 100:.1f}%)")
    print(f"Average actual project hours: {group['actual_hours'].mean():.2f}")
    print(f"Median actual project hours: {group['actual_hours'].median():.2f}")
    print(f"Median Predicted project hours: {group['predicted_hours'].median(): .2f}")
    print(f"Average absolute error: {group['abs_error_hours'].mean():.2f} hours")
    print(f"Average % error: {group['pct_error'].mean():.2f}%")
    print(f"Median % error: {group['pct_error'].median():.2f}%")

print("70% of the dataset results")

from sklearn.model_selection import train_test_split

df_train, df_unseen = train_test_split(df, test_size=0.30, random_state=42)
predictions_train = predict_hours(df_train)
predictions_unseen = predict_hours(df_unseen)

# =========================================================
# 70% of the dataset results (USED IN TRAINING)
# =========================================================
print("\n70% of the dataset results")

y_true_train = pd.to_numeric(df_train["Eng. AH"], errors="coerce")
y_pred_train = pd.to_numeric(predictions_train["predicted_Eng_AH"], errors="coerce")

mask_train = y_true_train.notna() & y_pred_train.notna()
y_true_train = y_true_train[mask_train]
y_pred_train = y_pred_train[mask_train]

error_train = np.abs(y_true_train - y_pred_train)
pct_error_train = np.where(y_true_train != 0, (error_train / y_true_train) * 100, np.nan)

summary_df_train = pd.DataFrame({
    "actual_hours": y_true_train.values,
    "predicted_hours": y_pred_train.values,
    "abs_error_hours": error_train.values,
    "pct_error": pct_error_train
})

bucket_definitions_train = [
    ("<1 hour", summary_df_train["abs_error_hours"] <= 1),
    ("1-2 hours", (summary_df_train["abs_error_hours"] > 1) & (summary_df_train["abs_error_hours"] <= 2)),
    ("2-3 hours", (summary_df_train["abs_error_hours"] > 2) & (summary_df_train["abs_error_hours"] <= 3)),
    ("3-7 hours", (summary_df_train["abs_error_hours"] > 3) & (summary_df_train["abs_error_hours"] <= 7)),
    (">7 hours", summary_df_train["abs_error_hours"] > 7),
]

print("\n===== Error Buckets with Project Size Context: 70% Training =====")
for label, condition in bucket_definitions_train:
    group = summary_df_train[condition]

    if len(group) == 0:
        print(f"{label}: 0 rows")
        continue

    print(f"\n{label}")
    print(f"Count: {len(group)} ({len(group) / len(summary_df_train) * 100:.1f}%)")
    print(f"Average actual project hours: {group['actual_hours'].mean():.2f}")
    print(f"Median actual project hours: {group['actual_hours'].median():.2f}")
    print(f"Average predicted project hours: {group['predicted_hours'].mean():.2f}")
    print(f"Median predicted project hours: {group['predicted_hours'].median():.2f}")


# =========================================================
# 30% of the dataset results (NOT USED IN TRAINING / UNSEEN)
# =========================================================
print("\n30% of the dataset results")

y_true_unseen = pd.to_numeric(df_unseen["Eng. AH"], errors="coerce")
y_pred_unseen = pd.to_numeric(predictions_unseen["predicted_Eng_AH"], errors="coerce")

mask_unseen = y_true_unseen.notna() & y_pred_unseen.notna()
y_true_unseen = y_true_unseen[mask_unseen]
y_pred_unseen = y_pred_unseen[mask_unseen]

error_unseen = np.abs(y_true_unseen - y_pred_unseen)
pct_error_unseen = np.where(y_true_unseen != 0, (error_unseen / y_true_unseen) * 100, np.nan)

summary_df_unseen = pd.DataFrame({
    "actual_hours": y_true_unseen.values,
    "predicted_hours": y_pred_unseen.values,
    "abs_error_hours": error_unseen.values,
    "pct_error": pct_error_unseen
})

bucket_definitions_unseen = [
    ("<1 hour", summary_df_unseen["abs_error_hours"] <= 1),
    ("1-2 hours", (summary_df_unseen["abs_error_hours"] > 1) & (summary_df_unseen["abs_error_hours"] <= 2)),
    ("2-3 hours", (summary_df_unseen["abs_error_hours"] > 2) & (summary_df_unseen["abs_error_hours"] <= 3)),
    ("3-7 hours", (summary_df_unseen["abs_error_hours"] > 3) & (summary_df_unseen["abs_error_hours"] <= 7)),
    (">7 hours", summary_df_unseen["abs_error_hours"] > 7),
]

print("\n===== Error Buckets with Project Size Context: 30% Unseen =====")
for label, condition in bucket_definitions_unseen:
    group = summary_df_unseen[condition]

    if len(group) == 0:
        print(f"{label}: 0 rows")
        continue

    print(f"\n{label}")
    print(f"Count: {len(group)} ({len(group) / len(summary_df_unseen) * 100:.1f}%)")
    print(f"Average actual project hours: {group['actual_hours'].mean():.2f}")
    print(f"Median actual project hours: {group['actual_hours'].median():.2f}")
    print(f"Average predicted project hours: {group['predicted_hours'].mean():.2f}")
    print(f"Median predicted project hours: {group['predicted_hours'].median():.2f}")

