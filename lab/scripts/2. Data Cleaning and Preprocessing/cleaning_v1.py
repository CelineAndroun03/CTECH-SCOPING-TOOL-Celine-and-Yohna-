import os
import json
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Raw file is in the SAME folder as this script
RAW_FILE = os.path.join(BASE_DIR, "Data v2.xlsx")

# Save outputs to lab/scripts/artifacts
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLEANED_XLSX = os.path.join(OUTPUT_DIR, "lab_cleaned_final.xlsx")
CLEANED_CSV = os.path.join(OUTPUT_DIR, "lab_cleaned_final.csv")
CONFIG_JSON = os.path.join(OUTPUT_DIR, "preprocessing_config.json")

TARGET_COL = "Lab. AH"
LOG_TARGET_COL = "Lab_AH_log"

# ============================================================
# SETTINGS
# ============================================================

SELECTED_FEATURES = [
    "standard_count",
    "total_CB_count",
    "total_test_count",
    "1 (60950-1)"
]

LEAKAGE_COLS = [
    "Lab. SH",
    "Eng. AH",
    "Eng. SH"
]

DROP_IF_EXISTS = [
    "Test/No Test"
]

# ============================================================
# HELPERS
# ============================================================

def safe_numeric(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def print_missing_summary(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("\nNo missing values.\n")
        return

    summary = pd.DataFrame({
        "Column": missing.index,
        "Number of Missing Values": missing.values,
        "% of Missing Values": (missing.values / len(df) * 100)
    })

    print("\nMissing values summary:\n")
    print(summary.to_string(index=False))

# ============================================================
# LOAD
# ============================================================

def load_data() -> pd.DataFrame:
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"Raw input file not found:\n{RAW_FILE}")

    df = pd.read_excel(RAW_FILE)
    print(f"\nLoaded raw data shape: {df.shape}")
    return df

# ============================================================
# MAIN CLEANING
# ============================================================

def clean_lab_data(df: pd.DataFrame):
    df = df.copy()

    print("\n==================================================")
    print("INITIAL CLEANING")
    print("==================================================")

    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped duplicates: {before - len(df)}")

    for col in DROP_IF_EXISTS:
        if col in df.columns:
            df = df.drop(columns=col)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    existing_leakage = [col for col in LEAKAGE_COLS if col in df.columns]
    print(f"Leakage columns removed from inputs: {existing_leakage}")

    for col in SELECTED_FEATURES + [TARGET_COL]:
        safe_numeric(df, col)

    print_missing_summary(df)

    df = df[df[TARGET_COL].notna()].copy()
    df = df[df[TARGET_COL] >= 0].copy()

    print(f"Shape after target filtering: {df.shape}")

    print("\n==================================================")
    print("NUMERIC PREPROCESSING")
    print("==================================================")

    if "standard_count" in df.columns:
        df["standard_count"] = df["standard_count"].fillna(0).astype(int)
    else:
        df["standard_count"] = 0

    if "total_CB_count" in df.columns:
        df["total_CB_count"] = df["total_CB_count"].fillna(0)
        df["total_CB_count"] = (df["total_CB_count"] > 0).astype(int)
    else:
        df["total_CB_count"] = 0

    if "total_test_count" in df.columns:
        df["total_test_count"] = df["total_test_count"].fillna(0)
    else:
        df["total_test_count"] = 0

    if "1 (60950-1)" in df.columns:
        df["1 (60950-1)"] = df["1 (60950-1)"].fillna(0).astype(int)
    else:
        df["1 (60950-1)"] = 0

    print("\n==================================================")
    print("FINAL FEATURE BUILD")
    print("==================================================")

    X = df[SELECTED_FEATURES].copy()
    X = X.fillna(0)

    y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0)
    y_log = np.log1p(y)

    final_df = pd.concat(
        [
            X,
            y.rename(TARGET_COL),
            y_log.rename(LOG_TARGET_COL)
        ],
        axis=1
    )

    print(f"Final cleaned shape: {final_df.shape}")
    print(f"Selected feature count: {len(SELECTED_FEATURES)}")

    print("\nSelected features:")
    for col in SELECTED_FEATURES:
        print(f"- {col}")

    config = {
        "selected_features": SELECTED_FEATURES,
        "numeric_cols": SELECTED_FEATURES,
        "total_cb_binary": True,
        "standard_count_int": True,
        "target_col": TARGET_COL,
        "log_target_col": LOG_TARGET_COL,
        "dropped_leakage_cols": existing_leakage,
        "feature_version": "LAB_4_INPUT_TEST"
    }

    return final_df, config

# ============================================================
# SAVE
# ============================================================

def save_outputs(final_df: pd.DataFrame, config: dict):
    final_df.to_excel(CLEANED_XLSX, index=False)
    final_df.to_csv(CLEANED_CSV, index=False)

    with open(CONFIG_JSON, "w") as f:
        json.dump(config, f, indent=4)

    print("\n==================================================")
    print("FILES SAVED")
    print("==================================================")
    print(f"Excel:  {CLEANED_XLSX}")
    print(f"CSV:    {CLEANED_CSV}")
    print(f"JSON:   {CONFIG_JSON}")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    raw_df = load_data()
    final_df, config = clean_lab_data(raw_df)
    save_outputs(final_df, config)
