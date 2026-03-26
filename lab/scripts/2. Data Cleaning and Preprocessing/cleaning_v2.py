
"""
Data Cleaning and Preprocessing & Feature Selection: Lab. AH

Purpose:
- Perform a robust Data Cleaning and Preprocessing & Feature Selection for target variable "Lab. AH".
- Save ALL outputs (printed text, key tables, all plots/figures) into a single Word (.docx) report.

Usage:
    python --input "1. Data.xlsx" --output "DCP&FS_Report.docx"

Notes:
- All figures are saved to ./reports/figures and embedded into the Word report with captions.
"""

# ==================================================
# 0. Imports & Global Configuration
# ==================================================
import argparse
import io
import os
from pathlib import Path
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import seaborn as sns
from scipy.stats import median_abs_deviation, zscore
from scipy import stats as sstats
from scipy.stats import mannwhitneyu, kruskal
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from docx import Document
from scipy.stats import fisher_exact, chi2_contingency
from docx.shared import Inches, Pt
from docx.enum.text import WD_BREAK
from contextlib import redirect_stdout
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
from scipy.stats import spearmanr
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# ==================================================
# 1. Import Data
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[2]
file_path = BASE_DIR / "scripts/2. Data Cleaning and Preprocessing/1. Data.xlsx"

data = pd.read_excel(file_path)

# ==================================================
# 2. Data Cleaning
# ==================================================

print("###############################################################")
print("####################### Data Cleaning #########################")
print("###############################################################")

# Remove Unecessary Features
data = data.drop(columns=["Lab. SH","field42 - Test/No Test"], errors="ignore")

# Remove Duplicates
data = data.drop_duplicates()
data["Lab. AH"] = np.log1p(data["Lab. AH"])

# Handle Missing Values
print("Missing values:")
missing_counts = data.isna().sum()
missing_pct = data.isna().mean() * 100
missing_table = (pd.DataFrame({"Column": data.columns,"Number of Missing Values": missing_counts.values,"% of Missing Values": missing_pct.values}).sort_values("% of Missing Values", ascending=False).reset_index(drop=True))
print(missing_table, "\n")

# ==================================================
# 3. Data Preprocessing
# ==================================================

print("###############################################################")
print("##################### Data Preprocessing ######################")
print("###############################################################\n")

# Handle Missing Values: Region
print("==================================================")
print("Handle Missing Values: Region")
print("==================================================\n")

data["Region_norm"] = data["Region"].astype(str).str.strip().str.upper()
data.loc[data["Region"].isna(), "Region_norm"] = "MISSING"
region_ohe = pd.get_dummies(data["Region_norm"], prefix="Region", drop_first=False)
data = pd.concat([data, region_ohe], axis=1)
data = data.drop(columns=["Region", "Missing_Region", "Region_MISSING", "Region_norm"], errors="ignore")
data = data.dropna()

# CCN_Data Hub
print("==================================================")
print("CCN_Data Hub")
print("==================================================\n")

TOPK = 10
HUB_COL = "CCN_Data Hub"
data[HUB_COL] = data[HUB_COL].astype(str).str.strip()
hub_counts = data[HUB_COL].value_counts()
hub_counts_no_zero = hub_counts.drop(labels=["0"], errors="ignore")
topK = set(hub_counts_no_zero.head(TOPK).index)
print(f"Top-{TOPK} hubs (excluding '0'):", sorted(topK))
data["CCN_Hub_topK"] = np.where(data[HUB_COL].isin(topK), data[HUB_COL], "OTHER")
ccn_ohe_topK = pd.get_dummies(data["CCN_Hub_topK"], prefix="CCN_TopK", drop_first=False)
data = pd.concat([data, ccn_ohe_topK], axis=1)
data = data.drop(columns=[HUB_COL, "CCN_Hub_topK"], errors="ignore")
print("Created CCN TopK OHE columns:", [c for c in data.columns if c.startswith("CCN_TopK_")])

# type_of_investigation
print("\n==================================================")
print("type_of_investigation")
print("====================================================\n")

data["TOI_norm"] = (data["type_of_investigation"].astype(str).str.strip().str.upper())
toi_ohe = pd.get_dummies(data["TOI_norm"],prefix="TOI",drop_first=False)
data = pd.concat([data, toi_ohe], axis=1)
data = data.drop(columns=["TOI_norm", "type_of_investigation"], errors="ignore")
print("Created TOI OHE columns:", list(toi_ohe.columns))

# Investigation_type
print("\n==================================================")
print("Investigation_type")
print("====================================================\n")

data["PT_norm"] = (data["Investigation_type"].astype(str).str.strip().str.upper())
it_ohe = pd.get_dummies(data["PT_norm"],prefix="PT",drop_first=False)
data = pd.concat([data, it_ohe], axis=1)
data = data.drop(columns=["PT_norm", "Investigation_type"], errors="ignore")
print("Created PT OHE columns:", list(it_ohe.columns))

# Drop columns with only 1 unique value
data = data.drop(columns = ["3 (60950-21)","5 (60950-23)"])

#unique_table = (data.nunique().reset_index().rename(columns={"index": "Coluna", 0: "Unique Values"}))
#print(unique_table)

# x = input("x")

# total_CB_count, total_test_count and standard_count
print("\n==================================================")
print("total_CB_count, total_test_count and standard_count")
print("====================================================\n")

data["total_CB_count"] = (data["total_CB_count"] > 0).astype(int)
data["total_test_count_Binary"] = (data["total_test_count"] > 0).astype(int)
data["standard_count"] = data["standard_count"].astype(int)

# Check unbalanced columns
print("\n==================================================")
print("Check unbalanced columns")
print("====================================================\n")

rows = []
candidates = []
for c in data.columns:
    if c == "Lab. AH":
        continue
    s = data[c]
    if pd.api.types.is_bool_dtype(s):
        candidates.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        u = pd.unique(s.dropna())
        if set(u).issubset({0,1}):
            candidates.append(c)
for c in candidates:
    s = data[c].astype(float)
    n = int(s.notna().sum())
    if n == 0:
        continue
    p1 = 100.0 * s.mean()
    p0 = 100.0 - p1
    near_constant = (p1 >= 95.0) or (p1 <= 5.0)
    if not near_constant:
        continue
    mask1 = (s == 1)
    mask0 = (s == 0)
    y1 = data.loc[mask1, "Lab. AH"].dropna().values
    y0 = data.loc[mask0, "Lab. AH"].dropna().values
    n1 = len(y1)
    n0 = len(y0)
    if n1 == 0 or n0 == 0:
        decision = "Drop"
        print(f"{decision:4s}  {c:40s}  %1={p1:6.2f}%   n1={n1:4d}   n0={n0:4d}   (no data in one group)")
        rows.append({
            "column": c, "%1": p1, "n1": n1, "n0": n0,
            "delta_mean_h": np.nan, "delta_median_h": np.nan,
            "p_mw": np.nan, "decision": decision
        })
        continue
    mean1 = np.mean(y1)
    mean0 = np.mean(y0)
    med1  = np.median(y1)
    med0  = np.median(y0)
    delta_mean   = mean1 - mean0
    delta_median = med1 - med0
    try:
        u_stat, p_mw = mannwhitneyu(y1, y0, alternative="two-sided")
    except Exception:
        p_mw = np.nan    
    if (n1 >= 20) and (abs(delta_mean) >= 0.20 or (not np.isnan(p_mw) and p_mw < 0.05)):
        decision = "Keep"
    else:
        decision = "Drop"
    print(f"{decision:4s}  {c:40s}  %1={p1:6.2f}%   n1={n1:4d}   "
          f"Δmean={delta_mean:+6.2f}h   Δmedian={delta_median:+6.2f}h   p(MW)={p_mw:.2e}")
    rows.append({
        "column": c, "%1": p1, "n1": n1, "n0": n0,
        "delta_mean_h": delta_mean, "delta_median_h": delta_median,
        "p_mw": p_mw, "decision": decision
    })
decisions = (
    pd.DataFrame(rows)
    .sort_values(["decision","%1"], ascending=[True, True])
    .reset_index(drop=True)
)
print("\nSummary of near-constant binary columns:")
print(decisions)


# Drop columns marked as "Drop" in decisions
cols_to_drop = decisions.loc[decisions["decision"] == "Drop", "column"].tolist()
print("Columns to drop:", cols_to_drop)
data = data.drop(columns=cols_to_drop, errors="ignore")
print("\nDropped", len(cols_to_drop), "columns.")

# Delete unkown columns
data = data.drop(columns = ["9 (Talk to Derek)","11 (Ask Derek if  IEC/EN 62368-3  covers this)",' 20 (Iteration of Tests ASK Derek)', '29 (Talk to Derek)','27 (Change Enc Func)'], errors="ignore")

# Check multicollinearity
print("\n==================================================")
print("Check multicollinearity")
print("====================================================\n")


bin_cols = []
for c in data.columns:
    if c == "Lab. AH":
        continue
    s = data[c]
    if pd.api.types.is_bool_dtype(s):
        bin_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        u = pd.unique(s.dropna())
        if set(u).issubset({0,1}):
            bin_cols.append(c)
if len(bin_cols) > 1:
    corr = data[bin_cols].corr(method="spearman").abs()

    high_pairs = []
    for i, a in enumerate(bin_cols):
        for j, b in enumerate(bin_cols):
            if j <= i:
                continue
            if corr.loc[a, b] >= 0.90:
                high_pairs.append((a, b, corr.loc[a, b]))

    if high_pairs:
        print("Highly redundant (>=0.90):")
        for a,b,v in high_pairs:
            print(f"{a} ~ {b}: {v:.2f}")
    else:
        print("No redundant binary pairs found (>=0.90).")
else:
    print("Not enough binary columns to compute correlations.")

data = data.drop(columns=["_60950_1_2ed_A2", "cUL", "CB"], errors="ignore")
data = data.drop_duplicates()
print()

data.to_excel("Data v2.xlsx")

# ==================================================
# 4. Feature Selection
# ==================================================
print("###############################################################")
print("##################### Feature Selection #######################")
print("###############################################################\n")
data = data.drop(columns=["total_test_count_Binary"])

# ============================================
# FEATURE SELECTION ONLY (A/B/C) — CORRECTED
# Assumes: 'data' exists and TARGET="Lab. AH" is log-transformed with np.log1p
# Selection on POSITIVES ONLY (Lab. AH > 0); scoring in REAL HOURS (np.expm1)
# Outputs:
#   - final_features.csv
#   - feature_selection_report.xlsx
#   - plots in ./plots_feature_selection
# ============================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.feature_selection import mutual_info_regression, RFECV
from sklearn.inspection import permutation_importance
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from scipy.stats import spearmanr

# Optional: XGBoost for Stage B (faster/more expressive than RF if available)
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# -----------------------------
# Configuration
# -----------------------------
TARGET = "Lab. AH"              # log-transformed (np.log1p) upstream
RANDOM_STATE = 42
N_SPLITS = 5
N_PERM_REPEATS = 8

PLOTS_DIR = "plots_feature_selection"
EXPORT_EXCEL = True
EXCEL_PATH = "feature_selection_report.xlsx"
FINAL_FEATURES_CSV = "final_features.csv"

# Set True to drop duplicates AFTER subsetting to positives & numeric features
DROP_DUPLICATES_POST_SUBSET = False

os.makedirs(PLOTS_DIR, exist_ok=True)

def savefig(path):
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

# Zero-gate "always keep" (never drop; needed downstream for zero-gate/eval)
CORE_TESTS = [
    "input_test", "heating_test", "abnormal_operations_test",
    "marking_test", "electric_strength_test"
]
COL_TOI4 = "TOI_4 - ADMINISTRATIVE NO TEST ANTICIPATED (REVISIONS REQUIRING ENGINEERING REVIEW)"
COL_IT4  = "IT_4 - DC DISTRIBUTION PANELS"

# Optional: domain-critical features to protect
ALWAYS_KEEP_DOMAIN = {
    # "standard_count", "total_test_count_Binary",
    # "IT Informational Test Report", "LCC_test", "pro_bond_test", "batteries_present",
}

ALWAYS_KEEP = set(CORE_TESTS + ["total_test_count", COL_TOI4, COL_IT4]) | set(ALWAYS_KEEP_DOMAIN)


def run_feature_selection(data: pd.DataFrame, target_col: str = TARGET, random_state: int = RANDOM_STATE):
    assert target_col in data.columns, f"Target '{target_col}' not found."

    # -----------------------------
    # Align selection with final regime:
    # - POSITIVES ONLY
    # - HOUR-SCALE SCORING (inverse of log target)
    # -----------------------------
    # 1) Build hours and positive mask
    y_log_full = data[target_col].astype(float).values
    y_hours_full = np.expm1(y_log_full)  # back to HOURS
    pos_mask = y_hours_full > 0

    # 2) Subset & keep numeric-only features
    X_full = data.drop(columns=[target_col], errors="ignore")
    X = X_full.loc[pos_mask].select_dtypes(include=[np.number]).copy()
    y_hours = y_hours_full[pos_mask].copy()

    # 3) Fill NaNs and (optionally) drop duplicates
    X = X.fillna(0)
    if DROP_DUPLICATES_POST_SUBSET:
        df_tmp = pd.concat([pd.Series(y_hours, name="_YH_"), X], axis=1).drop_duplicates()
        y_hours = df_tmp.pop("_YH_").values
        X = df_tmp.reset_index(drop=True)

    # 4) Drop constant columns
    non_constant_cols = [c for c in X.columns if pd.Series(X[c]).nunique(dropna=True) > 1]
    X = X[non_constant_cols]
    print(f"Initial feature count (positives only, numeric): {X.shape[1]}")

    # ---- Helper for MI: mark discrete/OHE
    def is_discrete_series(s, max_unique=10):
        if pd.api.types.is_bool_dtype(s):
            return True
        try:
            return s.nunique(dropna=True) <= max_unique
        except Exception:
            return False

    discrete_mask = np.array([is_discrete_series(X[c]) for c in X.columns], dtype=bool)

    # ==================================================
    # Stage A — Univariate (MI, |Spearman|, ShallowTrees)
    # ==================================================
    print("\n" + "="*50)
    print("Stage A — Univariate Screening (positives, hour-target)")
    print("="*50)

    # MI on HOURS (captures non-linear)
    mi = mutual_info_regression(
        X.values, y_hours,
        discrete_features=discrete_mask,
        random_state=random_state
    )
    mi_s = pd.Series(mi, index=X.columns, name="MI")

    # Spearman |rho| w/ HOURS
    rho_vals, pvals = [], []
    for c in X.columns:
        s = X[c]
        if s.nunique(dropna=True) < 2:
            rho, p = np.nan, np.nan
        else:
            rho, p = spearmanr(s, y_hours, nan_policy="omit")
        rho_vals.append(rho); pvals.append(p)
    spearman_df = pd.DataFrame({"|Spearman|": np.abs(rho_vals), "Spearman_p": pvals}, index=X.columns)

    # Shallow ExtraTrees (stable)
    shallow = ExtraTreesRegressor(
        n_estimators=400, max_depth=3, min_samples_leaf=2, n_jobs=-1, random_state=random_state
    )
    shallow.fit(X, y_hours)
    tree_imp = pd.Series(shallow.feature_importances_, index=X.columns, name="ShallowTreeImp")

    rank_df = pd.concat([mi_s, spearman_df["|Spearman|"], tree_imp], axis=1)

    # Composite score (z-normalized)
    z = (rank_df - rank_df.mean()) / (rank_df.std(ddof=0) + 1e-9)
    rank_df["CompositeScore"] = (
        z["MI"].fillna(0)*0.5 + z["|Spearman|"].fillna(0)*0.25 + z["ShallowTreeImp"].fillna(0)*0.25
    )

    # Keep top max(40, 60%) features by composite score (less aggressive)
    top_n = max(40, int(len(rank_df) * 0.60))
    mask_top = rank_df["CompositeScore"].rank(ascending=False, method="average").le(top_n)
    features_stageA = rank_df.index[mask_top].tolist()

    # Protect ALWAYS_KEEP
    for f in ALWAYS_KEEP:
        if f in X.columns and f not in features_stageA:
            features_stageA.append(f)

    print(f"[Stage A] Selected {len(features_stageA)} / {X.shape[1]} features for Stage B.")

    # Plots
    top_show = min(30, len(rank_df))
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)
    mi_s.sort_values(ascending=False).head(top_show).iloc[::-1].plot.barh(ax=axes[0], color="#3B82F6")
    axes[0].set_title("Top MI (Stage A, hours)")
    rank_df["|Spearman|"].sort_values(ascending=False).head(top_show).iloc[::-1].plot.barh(ax=axes[1], color="#10B981")
    axes[1].set_title("Top |Spearman| (Stage A, hours)")
    tree_imp.sort_values(ascending=False).head(top_show).iloc[::-1].plot.barh(ax=axes[2], color="#F59E0B")
    axes[2].set_title("Top Shallow Tree Importance (Stage A, hours)")
    savefig(os.path.join(PLOTS_DIR, "stageA_univariate.png"))

    # ==================================================
    # Stage B — Model-based importance with CV + null baseline (HOURS)
    # ==================================================
    print("\n" + "="*50)
    print("Stage B — Model-Based Importance with CV (positives, hour RMSE)")
    print("="*50)

    def make_base_estimator(seed=random_state):
        if XGB_AVAILABLE:
            return XGBRegressor(
                n_estimators=600, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=seed, tree_method="hist", n_jobs=-1
            )
        return RandomForestRegressor(
            n_estimators=600, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=seed
        )

    def neg_rmse_hours(y_true, y_pred):
        return -np.sqrt(mean_squared_error(y_true, y_pred))
    scorer = make_scorer(neg_rmse_hours, greater_is_better=True)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)

    X_B = X[features_stageA].values
    cols_B = features_stageA

    perm_list, model_imp_list, cv_scores, null_snapshots = [], [], [], []

    def permutation_null_baseline(estimator, X_va, y_va, scoring, n_repeats=3, seed=random_state):
        rng = np.random.RandomState(seed)
        null_means = []
        for r in range(n_repeats):
            y_shuf = rng.permutation(y_va)
            pi_null = permutation_importance(
                estimator, X_va, y_shuf, n_repeats=5, random_state=seed + r + 777, scoring=scoring
            )
            null_means.append(pi_null.importances_mean)
        null_means = np.vstack(null_means)
        return null_means.mean(axis=0), null_means.std(axis=0)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_B, y_hours), start=1):
        X_tr, X_va = X_B[tr_idx], X_B[va_idx]
        y_tr, y_va = y_hours[tr_idx], y_hours[va_idx]
        est = make_base_estimator(random_state + fold)
        est.fit(X_tr, y_tr)
        # Validation score (HOURS)
        y_va_pred = est.predict(X_va)
        cv_scores.append(neg_rmse_hours(y_va, y_va_pred))
        # Internal importance
        if hasattr(est, "feature_importances_"):
            model_imp_list.append(pd.Series(est.feature_importances_, index=cols_B))
        # Permutation importance (validation only)
        pi = permutation_importance(
            est, X_va, y_va,
            n_repeats=N_PERM_REPEATS, random_state=random_state + fold, scoring=scorer
        )
        perm_list.append(pd.Series(pi.importances_mean, index=cols_B))
        # Null baseline
        nm, ns = permutation_null_baseline(est, X_va, y_va, scoring=scorer, seed=random_state + fold)
        null_snapshots.append(pd.DataFrame({"null_mean": nm, "null_std": ns}, index=cols_B))

    print(f"[Stage B] Mean CV (neg RMSE in hours): {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    perm_mean = pd.concat(perm_list, axis=1).mean(axis=1)
    perm_std  = pd.concat(perm_list, axis=1).std(axis=1)
    stageB_df = pd.DataFrame({
        "PermImp_CV_mean": perm_mean,
        "PermImp_CV_std": perm_std
    }).sort_values("PermImp_CV_mean", ascending=False)

    if model_imp_list:
        model_imp_mean = pd.concat(model_imp_list, axis=1).mean(axis=1)
        stageB_df["ModelImp_mean"] = model_imp_mean.reindex(stageB_df.index).fillna(0.0)

    null_df_all = pd.concat(null_snapshots, axis=1)
    null_mean_avg = null_df_all.filter(like="null_mean").mean(axis=1)
    null_std_avg  = null_df_all.filter(like="null_std").mean(axis=1)

    # Less aggressive: 1σ over null; ensure at least top 50% or 35 features
    thresh = (null_mean_avg + 1.0 * null_std_avg).reindex(stageB_df.index).fillna(0.0)
    mask_strict = stageB_df["PermImp_CV_mean"] > thresh
    features_stageB = stageB_df.index[mask_strict].tolist()

    min_keep_stageB = max(35, int(0.50 * len(stageB_df)))
    if len(features_stageB) < min_keep_stageB:
        features_stageB = stageB_df.index[:min_keep_stageB].tolist()

    for f in ALWAYS_KEEP:
        if f in X.columns and f not in features_stageB:
            features_stageB.append(f)

    print(f"[Stage B] Selected {len(features_stageB)} / {len(cols_B)} features for Stage C.")

    # Plot Stage B
    top_show = min(30, len(stageB_df))
    plt.figure(figsize=(9, 10), constrained_layout=True)
    stageB_df["PermImp_CV_mean"].head(top_show).iloc[::-1].plot.barh(color="#6D28D9")
    plt.title("Permutation Importance on Validation (Stage B, hours)")
    savefig(os.path.join(PLOTS_DIR, "stageB_permutation.png"))

    # ==================================================
    # Stage C — RFECV (RepeatedKFold, HOURS)
    # ==================================================
    print("\n" + "="*50)
    print("Stage C — RFECV (positives, hour RMSE)")
    print("="*50)

    rfecv_est = RandomForestRegressor(
        n_estimators=600, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=random_state
    )
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=2, random_state=random_state)

    X_C = X[features_stageB].values
    cols_C = features_stageB

    rfecv = RFECV(
        estimator=rfecv_est,
        step=1,
        cv=rkf,
        scoring=make_scorer(lambda yt, yp: -np.sqrt(mean_squared_error(yt, yp)), greater_is_better=True),
        min_features_to_select=max(20, len([c for c in ALWAYS_KEEP if c in X.columns])),
        n_jobs=-1
    )
    rfecv.fit(X_C, y_hours)

    support = pd.Series(rfecv.support_, index=cols_C, name="keep")
    ranking = pd.Series(rfecv.ranking_, index=cols_C, name="rank")
    rfecv_df = pd.concat([support, ranking], axis=1).sort_values(["keep", "rank"], ascending=[False, True])

    # Conservative final set: RFECV ∪ top StageB ∪ ALWAYS_KEEP
    final_keep_rfecv = set(rfecv_df.index[rfecv_df["keep"]])
    final_union = final_keep_rfecv | set(stageB_df.index[:min_keep_stageB]) | {c for c in ALWAYS_KEEP if c in X.columns}
    features_stageC = sorted(final_union)

    print(f"[Stage C] Final selected features: {len(features_stageC)}")
    print("Examples:", features_stageC[:15], "..." if len(features_stageC) > 15 else "")

    # RFECV curve (best-effort)
    if hasattr(rfecv, "grid_scores_"):
        n_features_curve = np.arange(1, len(rfecv.grid_scores_) + 1)
        plt.figure(figsize=(8, 5), constrained_layout=True)
        plt.plot(n_features_curve, rfecv.grid_scores_, marker="o")
        plt.xlabel("Number of features selected")
        plt.ylabel("CV score (neg RMSE in hours)")
        plt.title("RFECV Performance Curve (Stage C, hours)")
        savefig(os.path.join(PLOTS_DIR, "stageC_rfecv_curve.png"))

    # Export audit + CSV
    if EXPORT_EXCEL:
        with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
            rank_df.sort_values("CompositeScore", ascending=False).to_excel(writer, sheet_name="StageA_Ranks")
            mi_s.sort_values(ascending=False).to_frame().to_excel(writer, sheet_name="StageA_MI")
            spearman_df.sort_values("|Spearman|", ascending=False).to_excel(writer, sheet_name="StageA_Spearman")
            tree_imp.sort_values(ascending=False).to_frame().to_excel(writer, sheet_name="StageA_TreeImp")

            stB_out = stageB_df.copy()
            stB_out["Null_mean"] = null_mean_avg.reindex(stB_out.index)
            stB_out["Null_std"]  = null_std_avg.reindex(stB_out.index)
            stB_out.to_excel(writer, sheet_name="StageB_Importance")

            rfecv_df.to_excel(writer, sheet_name="StageC_RFECV")
            pd.DataFrame({"Selected_Features": features_stageC}).to_excel(writer, sheet_name="Final_Features", index=False)
        print(f"Saved feature selection report to: {EXCEL_PATH}")

    # Save CSV list for the evaluation script
    pd.Series(features_stageC, name="Final_Features").to_csv(FINAL_FEATURES_CSV, index=False)
    print(f"Saved: {FINAL_FEATURES_CSV}")

    return features_stageC


# ---- Run if you want to execute immediately in your notebook/script ----
features_stageC = run_feature_selection(data, target_col=TARGET, random_state=RANDOM_STATE)

# ============================================================
# Dual evaluation: ALL numeric vs SELECTED features
#   - Target in `data[TARGET]` is already log-transformed upstream
#   - We invert to REAL HOURS for modeling/metrics
#   - Five models on positives; combined (positives + zeros=0)
#   - RUN B drops duplicates after narrowing to the selected columns
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

# -------------------------
# Zero-gate helpers (robust)
# -------------------------
CORE_TESTS = [
    "input_test",
    "heating_test",
    "abnormal_operations_test",
    "marking_test",
    "electric_strength_test"
]
COL_TOI4 = "TOI_4 - ADMINISTRATIVE NO TEST ANTICIPATED (REVISIONS REQUIRING ENGINEERING REVIEW)"
COL_IT4  = "IT_4 - DC DISTRIBUTION PANELS"
ALWAYS_KEEP_ZERO = set(CORE_TESTS + ["total_test_count", COL_TOI4, COL_IT4])

TARGET = "Lab. AH"
RANDOM_STATE = 42
FEATURES_CSV = "final_features.csv"

def ensure_zero_gate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure zero-gate columns exist; create zeros if missing."""
    for c in CORE_TESTS:
        if c not in df.columns:
            df[c] = 0
    if "total_test_count" not in df.columns:
        df["total_test_count"] = 0
    if COL_TOI4 not in df.columns:
        df[COL_TOI4] = 0
    if COL_IT4 not in df.columns:
        df[COL_IT4] = 0
    return df

def compute_zero_gate_mask(df: pd.DataFrame) -> pd.Series:
    df = ensure_zero_gate_columns(df)
    ttc0 = (df["total_test_count"] == 0)
    toi4 = (df[COL_TOI4] == 1)
    it4  = (df[COL_IT4] == 1)
    no_core = (df[CORE_TESTS].sum(axis=1) == 0)
    return (ttc0 | toi4 | (no_core & ttc0) | it4).astype(bool)

def metrics(name, y_true, y_pred):
    y_pred = np.clip(np.asarray(y_pred), a_min=0.0, a_max=None)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{name:>12s}  R2={r2:.3f}  RMSE={rmse:.3f}h  MAE={mae:.3f}h")
    return r2, rmse, mae

def run_two_split_eval(df_in: pd.DataFrame,
                       feature_cols,
                       scenario_name: str,
                       random_state: int = RANDOM_STATE,
                       drop_dups_after_subset: bool = False):
    """
    - If drop_dups_after_subset=True, first reduce to (feature_cols ∪ zero-keepers ∪ {TARGET}) then drop_duplicates()
    - Build zero_gate once on the (possibly deduped) df
    - Model POSITIVES in REAL HOURS (invert from log target)
    - Train 5 models; compute POS metrics and COMBINED metrics (positives + zeros=0)
    """
    print("\n" + "-"*50)
    print(f"{scenario_name}")
    print("-"*50)

    # -----------------------------------------
    # (Optional) Reduce columns & drop duplicates for this scenario
    # -----------------------------------------
    df = df_in.copy()
    keepers = {c for c in ALWAYS_KEEP_ZERO if c in df.columns}
    reduced_cols = sorted(set(feature_cols) | keepers | {TARGET})
    if drop_dups_after_subset:
        present_cols = [c for c in reduced_cols if c in df.columns]
        df = df[present_cols].copy().drop_duplicates()
        # Ensure zero-gate columns exist after the subset (create zeros if missing)
        df = ensure_zero_gate_columns(df)

    # -----------------------------------------
    # Zero-gate coverage on this dataset view
    # -----------------------------------------
    zero_gate = compute_zero_gate_mask(df)
    df_zero = df.loc[zero_gate].copy()      # predicted 0
    df_poscand = df.loc[~zero_gate].copy()  # to be modeled
    print(f"Zero-gate coverage: {len(df_zero)} rows ({100*len(df_zero)/len(df):.1f}%)")
    print(f"Modeling-candidates: {len(df_poscand)} rows ({100*len(df_poscand)/len(df):.1f}%)")

    # -----------------------------------------
    # POSITIVES in REAL HOURS (invert from stored log target)
    # -----------------------------------------
    y_pos_hours_all = np.expm1(df_poscand[TARGET].astype(float).fillna(0).values)
    y_pos_hours_all = np.clip(y_pos_hours_all, 0, None)

    pos_mask = (y_pos_hours_all > 0)
    df_pos = df_poscand.loc[pos_mask].copy()
    y = y_pos_hours_all[pos_mask]
    print(f"Positives to model: {len(df_pos)}")

    # -----------------------------------------
    # Build X: requested features (+ zero-keepers), numeric only, cast bools
    # -----------------------------------------
    feat_cols = [c for c in sorted(set(feature_cols) | keepers)
                 if c in df_pos.columns and pd.api.types.is_numeric_dtype(df_pos[c])]
    if len(feat_cols) == 0:
        print("[WARN] Feature list empty; falling back to ALL numeric (+ zero-keepers).")
        all_numeric = [c for c in df_pos.columns if pd.api.types.is_numeric_dtype(df_pos[c]) and c != TARGET]
        feat_cols = sorted(set(all_numeric) | keepers)

    for c in feat_cols:
        if pd.api.types.is_bool_dtype(df_pos[c]):
            df_pos[c] = df_pos[c].astype(np.int8)

    X = df_pos[feat_cols].copy()
    print(f"Using {len(feat_cols)} features: {feat_cols[:20]}{' ...' if len(feat_cols) > 20 else ''}")

    # -----------------------------------------
    # Splits (stratify on HOURS)
    # -----------------------------------------
    percentiles = [0, 70, 85, 93, 97, 100]
    bin_edges = np.percentile(y, percentiles)
    bin_edges = np.unique(bin_edges)
    while len(bin_edges) < 3:
        bin_edges = np.append(bin_edges, bin_edges[-1] + 1)
    bin_edges[-1] = bin_edges[-1] + 1e-6
    y_bins = np.digitize(y, bin_edges)

    X_train, X_temp, y_train, y_temp, bins_train, bins_temp = train_test_split(
        X, y, y_bins, test_size=0.40, random_state=random_state, stratify=y_bins
    )
    X_val, X_test, y_val, y_test, _, _ = train_test_split(
        X_temp, y_temp, bins_temp, test_size=0.50, random_state=random_state, stratify=bins_temp
    )
    print("Positives Split → Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # -----------------------------------------
    # Zeros split for COMBINED metrics (HOURS)
    # -----------------------------------------
    if len(df_zero) > 0:
        z_hours = np.expm1(df_zero[TARGET].astype(float).fillna(0).values)
        z_hours = np.clip(z_hours, 0, None)
        Z_train, Z_temp, z_train, z_temp = train_test_split(
            df_zero, z_hours, test_size=0.40, random_state=random_state
        )
        Z_val, Z_test, z_val, z_test = train_test_split(
            Z_temp, z_temp, test_size=0.50, random_state=random_state
        )
    else:
        Z_train = Z_val = Z_test = pd.DataFrame(columns=X.columns)
        z_train = z_val = z_test = np.array([])

    print("Zeros Split → Train:", Z_train.shape, "Val:", Z_val.shape, "Test:", Z_test.shape)

    # ============================================================
    # Models on POSITIVES (hours)
    # ============================================================
    results = {}

    # MODEL 1 — XGB Tweedie (hours)
    print("\n===================== MODEL 1: XGB Tweedie (positives) =====================")
    model1 = XGBRegressor(
        objective="reg:tweedie", tweedie_variance_power=1.3,
        n_estimators=3000, learning_rate=0.02,
        max_depth=3, min_child_weight=8,
        subsample=0.7, colsample_bytree=0.7,
        reg_lambda=2.0, random_state=random_state,
        tree_method="hist", eval_metric="rmse",
        early_stopping_rounds=200
    )
    model1.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    p1_tr = model1.predict(X_train); p1_va = model1.predict(X_val); p1_te = model1.predict(X_test)
    results["XGB_Tweedie"] = {"p_tr": p1_tr, "p_va": p1_va, "p_te": p1_te}
    metrics("Pos-Train", y_train, p1_tr); metrics("Pos-Val", y_val, p1_va); metrics("Pos-Test", y_test, p1_te)

    # MODEL 2 — XGB log1p (train on log(hours), report hours)
    print("\n===================== MODEL 2: XGB log1p (positives) =====================")
    y_train_log = np.log1p(y_train); y_val_log = np.log1p(y_val)
    model2 = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=3000, learning_rate=0.03,
        max_depth=3, min_child_weight=8,
        subsample=0.7, colsample_bytree=0.7,
        reg_lambda=2.0, random_state=random_state,
        tree_method="hist", eval_metric="rmse",
        early_stopping_rounds=200
    )
    model2.fit(X_train, y_train_log, eval_set=[(X_train, y_train_log), (X_val, y_val_log)], verbose=False)
    p2_tr = np.expm1(model2.predict(X_train))
    p2_va = np.expm1(model2.predict(X_val))
    p2_te = np.expm1(model2.predict(X_test))
    results["XGB_log1p"] = {"p_tr": p2_tr, "p_va": p2_va, "p_te": p2_te}
    metrics("Pos-Train", y_train, p2_tr); metrics("Pos-Val", y_val, p2_va); metrics("Pos-Test", y_test, p2_te)

    # # MODEL 3 — LightGBM (hours)
    # print("\n===================== MODEL 3: LightGBM (positives) =====================")
    # model3 = LGBMRegressor(
    #     n_estimators=3000, learning_rate=0.03, max_depth=-1,
    #     subsample=0.7, colsample_bytree=0.7, reg_lambda=2.0,
    #     random_state=random_state
    #)
    # model3.fit(X_train, y_train)
    # p3_tr = model3.predict(X_train); p3_va = model3.predict(X_val); p3_te = model3.predict(X_test)
    # results["LGBM"] = {"p_tr": p3_tr, "p_va": p3_va, "p_te": p3_te}
    # metrics("Pos-Train", y_train, p3_tr); metrics("Pos-Val", y_val, p3_va); metrics("Pos-Test", y_test, p3_te)

    # MODEL 4 — RandomForest (hours)
    print("\n===================== MODEL 4: RandomForest (positives) =====================")
    model4 = RandomForestRegressor(
        n_estimators=600, max_depth=None, min_samples_leaf=2,
        random_state=random_state, n_jobs=-1
    )
    model4.fit(X_train, y_train)
    p4_tr = model4.predict(X_train); p4_va = model4.predict(X_val); p4_te = model4.predict(X_test)
    results["RF"] = {"p_tr": p4_tr, "p_va": p4_va, "p_te": p4_te}
    metrics("Pos-Train", y_train, p4_tr); metrics("Pos-Val", y_val, p4_va); metrics("Pos-Test", y_test, p4_te)

    # MODEL 5 — ElasticNet (hours)
    print("\n===================== MODEL 5: ElasticNet (positives) =====================")
    model5 = ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=random_state)
    model5.fit(X_train, y_train)
    p5_tr = model5.predict(X_train); p5_va = model5.predict(X_val); p5_te = model5.predict(X_test)
    results["ElasticNet"] = {"p_tr": p5_tr, "p_va": p5_va, "p_te": p5_te}
    metrics("Pos-Train", y_train, p5_tr); metrics("Pos-Val", y_val, p5_va); metrics("Pos-Test", y_test, p5_te)

    # -----------------------------------------
    # Combined metrics (positives + zeros=0) in HOURS
    # -----------------------------------------
    chosen = "XGB_log1p" if "XGB_log1p" in results else (
             "XGB_Tweedie" if "XGB_Tweedie" in results else (
             "LGBM" if "LGBM" in results else "RF"))
    pos_pred = {"Train": results[chosen]["p_tr"], "Val": results[chosen]["p_va"], "Test": results[chosen]["p_te"]}
    pos_true = {"Train": y_train, "Val": y_val, "Test": y_test}

    print("\n===================== COMBINED (positives + zeros=0) =====================")
    for split, yp_pos, yt_pos, z in [("Val", pos_pred["Val"], pos_true["Val"], z_val),
                                     ("Test", pos_pred["Test"], pos_true["Test"], z_test),
                                     ("Train", pos_pred["Train"], pos_true["Train"], z_train)]:
        y_true_comb = np.concatenate([yt_pos, z]) if len(z) > 0 else yt_pos
        y_pred_comb = np.concatenate([yp_pos, np.zeros_like(z)]) if len(z) > 0 else yp_pos
        metrics(f"Comb-{split}", y_true_comb, y_pred_comb)

# ============================================================
# RUN A — ALL numeric features (except target)
# ============================================================
all_numeric_feats = [c for c in data.columns if c != TARGET and pd.api.types.is_numeric_dtype(data[c])]
run_two_split_eval(data, feature_cols=all_numeric_feats, scenario_name="RUN A — Using ALL numeric features", random_state=RANDOM_STATE, drop_dups_after_subset=False)

# ============================================================
# RUN B — SELECTED features (+ zero-keepers) with post-subset DROP DUPLICATES
# ============================================================
if os.path.exists(FEATURES_CSV):
    selected_features = pd.read_csv(FEATURES_CSV).iloc[:, 0].dropna().astype(str).tolist()
    print(f"\nLoaded {len(selected_features)} selected features from {FEATURES_CSV}")
elif "features_stageC" in globals():
    selected_features = list(features_stageC)
    print(f"\nUsing {len(selected_features)} selected features from variable 'features_stageC'")
else:
    raise FileNotFoundError(f"'{FEATURES_CSV}' not found and 'features_stageC' not in memory. Run the Feature Selection script first.")

run_two_split_eval(data, feature_cols=selected_features, scenario_name="RUN B — Using SELECTED features (+ drop_duplicates post-subset)", random_state=RANDOM_STATE, drop_dups_after_subset=True)




