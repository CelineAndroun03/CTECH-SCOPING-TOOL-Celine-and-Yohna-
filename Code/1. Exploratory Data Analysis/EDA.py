# ##################################################################
#     #1) Load the Excel dataset
# ##################################################################

# 1: Import Libraries
from scipy.stats import skew
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pandas.api.types as ptypes
import warnings
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# 2: Download Data
data = pd.read_excel("Final_data_Ctech.xlsx")
print("Shape:", data.shape) #Prints (rows,columns)


#3- ############Delete DUPLICATES (DONE)############################
print("total rows:", len(data))
print("Exact duplicate rows:", data.duplicated().sum())

#4-###############Check for missing values within each column##############################################

missing_Counts = data.isna().sum()
missing_Percent = (data.isna().mean()* 100)

missing_summary = pd.DataFrame({
    "Missing_Count": data.isna().sum(),
    "Missing_Percent": data.isna().mean() * 100
})

missing_summary = missing_summary.sort_values(
    by="Missing_Percent",
    ascending=False
)

print(missing_summary[missing_summary["Missing_Count"] > 0])

############################REGION Missing values test##############################################
print("\nRegion Kruskal Test")
subset = data.dropna(subset=["Region", "Eng. AH"])

groups = [group["Eng. AH"].values
          for name, group in subset.groupby("Region")]

print(kruskal(*groups))

############################Investigation Type missing values test##############################################
print("\nInvestigation type Kruskal Test")
subset = data.dropna(subset=["Investigation_type", "Eng. AH"])

groups = [group["Eng. AH"].values
          for name, group in subset.groupby("Investigation_type")]

print(kruskal(*groups))

############################Derek 22 missing values test##############################################
# print("\n22 Derek Kruskal Test")
# subset = data.dropna(subset=["22 (Derek Understand when used)", "Eng. AH"])

# groups = [group["Eng. AH"].values
#           for name, group in subset.groupby("22 (Derek Understand when used)")]

# print(kruskal(*groups))

# data["Derek_missing"] = data["22 (Derek Understand when used)"].isna()

# subset = data.dropna(subset=["Eng. AH"])

# print("\nAre projects with missing values different from projects with non-missing values?")
# groups = [group["Eng. AH"].values
#           for name, group in subset.groupby("Derek_missing")]

# print(kruskal(*groups))


####################################################################
# -----------------------------
# 0) Clean target (Eng. AH)
# -----------------------------
df = data[['Region', 'Eng. AH']].copy()

# Convert Eng. AH to numeric (turns '#N/A' or text into NaN)
df['Eng. AH'] = pd.to_numeric(df['Eng. AH'], errors='coerce')

# Drop rows where target is missing after conversion
df = df.dropna(subset=['Eng. AH'])

# =========================================================
# TEST 1: ONE-HOT ENCODING (REGION)
# =========================================================
print("\n")
print("Region Encoding Test")
X_onehot = pd.get_dummies(df['Region'].fillna('Missing'), drop_first=True)

# Force numeric (important for statsmodels)
X_onehot = X_onehot.astype(float)

X_onehot = sm.add_constant(X_onehot)

model_onehot = sm.OLS(df['Eng. AH'].astype(float), X_onehot).fit()
print("One-Hot Encoding R2:", model_onehot.rsquared)

# =========================================================
# TEST 2: LABEL ENCODING (REGION)
# =========================================================

df_label = df.copy()
df_label['Region'] = df_label['Region'].fillna('Missing')

le = LabelEncoder()
df_label['Region_encoded'] = le.fit_transform(df_label['Region'])

X_label = sm.add_constant(df_label[['Region_encoded']].astype(float))

model_label = sm.OLS(df_label['Eng. AH'].astype(float), X_label).fit()
print("Label Encoding R2:", model_label.rsquared)


#########################################################################
#################### Nuno cleaning block ################################

model_data = data.copy()
""" Nuno: The following Code shows the thing you should do from the beggining.
        - "Eng. SH", "Lab. SH", "Lab. AH" droped because they are out of our scope and/or are variables that we will not have in prod to make predictions
    It is also crucial to remove the duplicated rows.
    Also, from the Exploratory Data Analysis we saw that we need to log transform the target.
"""
#model_data = model_data.drop(columns=["Eng. SH", "Lab. SH", "Lab. AH"])
model_data = model_data.drop_duplicates()
model_data["Eng. AH"] = np.log(model_data["Eng. AH"])

#1 Drop Derek 22 Column
model_data = model_data.drop(columns=["22 (Derek Understand when used)"], errors="ignore")

if "22 (Derek Understand when used)" not in model_data.columns:
    print("SUCCESS: Column Deleted")
else:
    print("Column still exists")

#2 Use One-Hot Encoding for Region 
model_data["Region"] = model_data["Region"].fillna("Missing")
region_ohe = pd.get_dummies(model_data["Region"], prefix="Region").astype(int)
model_data = pd.concat ( [model_data, region_ohe], axis=1)
model_data = model_data.drop(columns=["Region"])
model_data = model_data.drop(columns=["Region_Missing"], errors="ignore")
print(model_data.filter(like="Region").head())
Region_cols = ["Region_AMERICAS", "Region_ASIA"]
print(model_data[Region_cols].sum())

"""
Nuno: After dealing with Region (15% of missing values), we can drop the missing values from the other columns as they have little cases
and we can't with certainty handle them. There are approaches to handle them but we will just delete them in this case.
"""
model_data = model_data.dropna()

#3 Use One-Hot Encoding for Investigation_type
model_data["Investigation_type"] = model_data["Investigation_type"].fillna("Missing")
inv_ohe = pd.get_dummies(model_data["Investigation_type"], prefix="Investigation_type").astype(int)
model_data = pd.concat ( [model_data, inv_ohe], axis=1)
model_data = model_data.drop(columns=["Investigation_type"])
#model_data = model_data.drop(columns=["Investigation_type_Missing"], errors="ignore")
print(model_data.filter(like="Investigation_type").head())
inv_cols = model_data.filter(like="Investigation_type_").columns
print("\n")
print(model_data[inv_cols].sum(axis=1).value_counts()) #64 missing values, dropping it
model_data = model_data[model_data[inv_cols].sum(axis=1) > 0] #Drop blank Cells
model_data[inv_cols].sum(axis=1).value_counts() #Confirm it worked

#4 Use One-Hot Encoding for type_of_investigation  
model_data["type_of_investigation"] = model_data["type_of_investigation"].fillna("Missing")
TOI_ohe = pd.get_dummies(model_data["type_of_investigation"], prefix="type_of_investigation").astype(int)
model_data = pd.concat ( [model_data, TOI_ohe], axis=1)
model_data = model_data.drop(columns=["type_of_investigation"], errors="ignore")
#model_data = model_data.drop(columns=["type_of_investigation_Missing"]) 
print(model_data.filter(like="type_of_investigation").head())
TOI_cols = model_data.filter(like="type_of_investigation_").columns
print(model_data[TOI_cols].sum())
print("\n")
print(model_data[TOI_cols].sum(axis=1).value_counts())

#5 Converting counts into meaningful features

#total_CB_count -> binary (0/1)
model_data["total_CB_count"] = pd.to_numeric(model_data["total_CB_count"], errors="coerce")
model_data["total_CB_count_Binary"] = (model_data["total_CB_count"].fillna(0) > 0).astype(int)

#total_test_count_Binary -> binary (0/1)
model_data["total_test_count"] = pd.to_numeric(model_data["total_test_count"], errors="coerce")
model_data["total_test_count_Binary"] = (model_data["total_test_count"].fillna(0) > 0).astype(int)

#standard_count -> int (update dropping this)
model_data["standard_count"] = pd.to_numeric(model_data["standard_count"], errors="coerce")
model_data["standard_count_Binary"] = model_data["standard_count"].fillna(0).astype(int)
model_data = model_data.drop(
    columns=["total_CB_count_Binary", "total_test_count_Binary", "standard_count_Binary"],errors="ignore"
)


#CCN_Data Hub Top 10 + OTHER + OHE
TOPK = 10
HUB_COL = "CCN_Data Hub"

model_data[HUB_COL] = model_data[HUB_COL].fillna("Missing").astype(str).str.strip()

hub_counts = model_data[HUB_COL].value_counts()
hub_counts = hub_counts.drop(labels=["0"], errors="ignore")

top10 = set(hub_counts.head(TOPK).index)

model_data["CCN_Hub_top10"] = np.where(model_data[HUB_COL].isin(top10),
                                      model_data[HUB_COL],
                                      "OTHER")

ccn_ohe = pd.get_dummies(model_data["CCN_Hub_top10"], prefix="CCN_Top10").astype(int)
model_data = pd.concat([model_data, ccn_ohe], axis=1)

model_data = model_data.drop(columns=[HUB_COL, "CCN_Hub_top10"], errors="ignore")

print(" Step 5 complete")

#Quick Check (binary columns)
print(model_data["total_CB_count"].describe())
print(model_data["total_test_count"].describe())
print(model_data["standard_count"].describe())


#Check CCN Top10 columns
print([c for c in model_data.columns if c.startswith("CCN_Top10_")])
print(model_data.shape)
print("\n")

# STEP 6: Drop columns with only one unique value
single_value_cols = []

for col in model_data.columns:
    unique_vals = model_data[col].nunique(dropna=False)
    if unique_vals <= 1:
        single_value_cols.append(col)
        print(f"{col}  --> unique values: {unique_vals}")

print("\nTotal columns to drop:", len(single_value_cols)) #four of them

"""
Drop columns with just one value
"""
model_data = model_data.drop(columns=single_value_cols, errors="ignore")

#Columns where 90% or more share the same value
THRESH = 0.95  # 90%

results = []

n_rows = len(model_data)

for col in model_data.columns:
    vc = model_data[col].value_counts(dropna=False)
    top_value = vc.index[0]
    top_count = int(vc.iloc[0])
    top_pct = top_count / n_rows

    if top_pct >= THRESH:
        results.append({
            "column": col,
            "top_value": top_value,
            "top_count": top_count,
            "top_percent": round(top_pct * 100, 2),
            "unique_values": int(model_data[col].nunique(dropna=False))
        })

# sort: most “constant” first
results_df = pd.DataFrame(results).sort_values("top_percent", ascending=False)

print(f"\nColumns with >= {int(THRESH*100)}% same value: {len(results_df)}\n")
if len(results_df) > 0:
    # show all rows (or change to .head(50))
    print(results_df.to_string(index=False))
else:
    print("None found.")

# Optional: save to Excel for easy filtering
results_df.to_excel("near_constant_columns_95pct.xlsx", index=False)
print("\nSaved: near_constant_columns_95pct.xlsx")

# Check unbalanced columns
print("\n==================================================")
print("Check unbalanced columns")
print("====================================================\n")

rows = []
candidates = []
for c in model_data.columns:
    if c == "Eng. AH":
        continue
    s = model_data[c]
    if pd.api.types.is_bool_dtype(s):
        candidates.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        u = pd.unique(s.dropna())
        if set(u).issubset({0,1}):
            candidates.append(c)
for c in candidates:
    s = model_data[c].astype(float)
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
    y1 = model_data.loc[mask1, "Eng. AH"].dropna().values
    y0 = model_data.loc[mask0, "Eng. AH"].dropna().values
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

"""
The first test only checks whether a column is almost constant (e.g., 99% of values are the same). This tells you the column has very low variability,
but it does not tell you whether the rare cases actually have an impact on the target. The second test is better because it goes further: it checks not
only the imbalance but also the sample size in the minority group and the real effect of the feature on the target (difference in means/medians and
statistical significance). This means we only keep rare flags that actually change the target in a meaningful way, and we drop the ones that are
rare and useless, improving model quality and reducing noise.
"""
drop_cols = [
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
    "type_of_investigation_5 - Administrative CB review"
]
model_data = model_data.drop(columns=drop_cols, errors="ignore")


drop_cols = ["9 (Talk to Derek)","4 (Annex Y)","11 (Ask Derek if  IEC/EN 62368-3  covers this)","34 (may be duplicate - check)",
             "30 (Talk to Derek)"," 20 (Iteration of Tests ASK Derek)","29 (Talk to Derek)","27 (Change Enc Func)"]
model_data = model_data.drop(columns=drop_cols, errors="ignore")
print("Number of unique values (per column):")
print(model_data.nunique(), "\n")


# Check multicollinearity
print("\n==================================================")
print("Check multicollinearity")
print("====================================================\n")

bin_cols = []
for c in model_data.columns:
    if c == "Eng. AH":
        continue
    s = model_data[c]
    if pd.api.types.is_bool_dtype(s):
        bin_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        u = pd.unique(s.dropna())
        if set(u).issubset({0,1}):
            bin_cols.append(c)
if len(bin_cols) > 1:
    corr = model_data[bin_cols].corr(method="spearman").abs()

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

model_data = model_data.drop(columns=["total_CB_count_Binary", "_60950_1_2ed_A2", "capacitance_discharge_test"], errors="ignore")
model_data = model_data.drop_duplicates()


# # Compare touch_current_single_test & capacitance_discharge_test correlation with Eng. AH
# touch_corr = abs(spearmanr(
#     model_data["\ntouch_current_single_test"],
#     model_data["Eng. AH"]
# )[0])

# cap_corr = abs(spearmanr(
#     model_data["capacitance_discharge_test"],
#     model_data["Eng. AH"]
# )[0])

# print(f"Touch Current vs Eng AH: {touch_corr:.4f}")
# print(f"Capacitance Discharge vs Eng AH: {cap_corr:.4f}")
# #touch_current_single_test is slightly stronger

###############################
########## Next Steps BELOW #########
###############################



###############################
########## STAGE A #########
###############################

#PART 1: MUTUAL INFORMATION

target = "Eng. AH"

# Separate X and y
y = model_data[target].astype(float)
X = model_data.drop(columns=[target], errors="ignore")

# Keep only numeric columns (important)
X = X.select_dtypes(include=[np.number]).copy()

# Fill any remaining NaNs just in case
X = X.fillna(0)

# ------------------------------------------------------------
# Identify discrete features (binary or low-cardinality)
# ------------------------------------------------------------
discrete_mask = []

for col in X.columns:
    unique_vals = pd.unique(X[col].dropna())
    
    # Binary columns OR very low-cardinality columns treated as discrete
    if set(unique_vals).issubset({0, 1}) or len(unique_vals) <= 5:
        discrete_mask.append(True)
    else:
        discrete_mask.append(False)

# ------------------------------------------------------------
# Compute Mutual Information (properly handling discrete vars)
# ------------------------------------------------------------
mi_scores = mutual_info_regression(
    X,
    y,
    discrete_features=np.array(discrete_mask),
    random_state=42
)

# Put into DataFrame
mi_df = pd.DataFrame({
    "Feature": X.columns,
    "MI_Score": mi_scores
})

# Sort highest first
mi_df = mi_df.sort_values("MI_Score", ascending=False)

print("\nTop 20 Features by Mutual Information:\n")
print(mi_df.head(20))


#PART 2: Spearman Correlation
# Compute Spearman correlation for each feature vs target
spearman_scores = X.apply(lambda col: col.corr(y, method="spearman"))

# Take absolute value (we care about strength, not direction yet)
spearman_df = pd.DataFrame({
    "Feature": spearman_scores.index,
    "Spearman": spearman_scores.values
})

spearman_df["Abs_Spearman"] = spearman_df["Spearman"].abs()

# Sort by absolute strength
spearman_df = spearman_df.sort_values("Abs_Spearman", ascending=False)

print("\n\nTop 20 Features by Spearman:\n")
print(spearman_df.head(20))


#PART 3: Shallow Tree Importance
# Use same X and y from earlier
tree = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=3,
        random_state=42,
        n_jobs=-1
    )

tree.fit(X, y)

# Get importance scores
tree_importance = pd.DataFrame({
    "Feature": X.columns,
    "Tree_Importance": tree.feature_importances_
}).sort_values("Tree_Importance", ascending=False)
print(tree_importance.head(20))

# Sort highest first
tree_importance = tree_importance.sort_values(
    "Tree_Importance",
    ascending=False
)

print("\n\nTop 20 Features by Shallow Tree:\n")
print(tree_importance.head(20))

#PART 4(EXTRA STEP): COMPARE ALL 3

print("\n\n==================================================")
print("Top 20 Features (MI + Spearman + Tree aligned)")
print("====================================================\n")
TOP_N = 21
NAME_MAX = 50


# Align safely
mi_tmp   = mi_df.set_index("Feature")[["MI_Score"]]
sp_tmp   = spearman_df.set_index("Feature")[["Abs_Spearman"]]
tree_tmp = tree_importance.set_index("Feature")[["Tree_Importance"]]

comparison = (
    mi_tmp.join(sp_tmp, how="outer")
          .join(tree_tmp, how="outer")
          .fillna(0)
)

# Sort by MI primarily
comparison = comparison.sort_values(
    by=["MI_Score", "Abs_Spearman", "Tree_Importance"],
    ascending=False
).head(TOP_N).reset_index()

# Shorten long feature names
comparison["Feature"] = (
    comparison["Feature"].astype(str)
    .str.replace("type_of_investigation_", "TOI_", regex=False)
    .str.replace("Investigation_type_", "INV_", regex=False)
    .str.replace("CCN_Top10_", "CCN_", regex=False)
    .str.slice(0, NAME_MAX)
)

# Rename columns cleanly
comparison = comparison.rename(columns={
    "MI_Score": "MI",
    "Abs_Spearman": "Spearman",
    "Tree_Importance": "Tree"
})

# Round for presentation
comparison["MI"] = comparison["MI"].round(4)
comparison["Spearman"] = comparison["Spearman"].round(4)
comparison["Tree"] = comparison["Tree"].round(4)

print(comparison[["Feature", "MI", "Spearman", "Tree"]].to_string(index=False))


#######################################################################################################################################################
print("\n\nStage B starts Here\n")
###############################
########## STAGE B #########
###############################
X_model = X.copy()
y_model = y.copy()  # LOG(hours)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ---------- Baseline RMSE (REAL HOURS) ----------
y_hours = np.exp(y_model)
baseline_pred_hours = y_hours.mean()
baseline_rmse_hours = np.sqrt(mean_squared_error(y_hours, np.full_like(y_hours, baseline_pred_hours)))
print("\nBaseline RMSE (hours):", round(float(baseline_rmse_hours), 2))

# ---------- Custom scorer: NEG RMSE in HOURS ----------
def neg_rmse_hours(y_true_log, y_pred_log):
    y_true_h = np.exp(y_true_log)
    y_pred_h = np.exp(y_pred_log)
    rmse_h = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
    return -rmse_h

perm_scorer_hours = make_scorer(neg_rmse_hours, greater_is_better=True)

rmse_hours_scores = []
gain_importance_accumulator = np.zeros(X_model.shape[1])

perm_real_accumulator = np.zeros(X_model.shape[1])
perm_null_accumulator = np.zeros(X_model.shape[1])

for fold, (train_idx, val_idx) in enumerate(kf.split(X_model), start=1):
    X_train, X_val = X_model.iloc[train_idx], X_model.iloc[val_idx]
    y_train, y_val = y_model.iloc[train_idx], y_model.iloc[val_idx]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # ---------- RMSE in REAL HOURS ----------
    y_pred_log = model.predict(X_val)
    y_val_hours  = np.exp(y_val)
    y_pred_hours = np.exp(y_pred_log)

    rmse_h = np.sqrt(mean_squared_error(y_val_hours, y_pred_hours))
    rmse_hours_scores.append(rmse_h)
    print(f"Fold {fold} RMSE (hours): {rmse_h:.2f}")

    # ---------- Gain importance ----------
    gain_importance_accumulator += model.feature_importances_

    # ---------- Permutation importance (REAL HOURS scorer) ----------
    perm_real = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=5,
        random_state=42,
        scoring=perm_scorer_hours
    )
    perm_real_accumulator += perm_real.importances_mean

    # ---------- Null baseline: shuffled target ----------
    y_val_shuff = y_val.sample(frac=1.0, random_state=42).reset_index(drop=True)
    X_val_reset = X_val.reset_index(drop=True)

    perm_null = permutation_importance(
        model,
        X_val_reset,
        y_val_shuff,
        n_repeats=5,
        random_state=42,
        scoring=perm_scorer_hours
    )
    perm_null_accumulator += perm_null.importances_mean


avg_rmse_hours = float(np.mean(rmse_hours_scores))
avg_gain = gain_importance_accumulator / kf.get_n_splits()

avg_perm_real = perm_real_accumulator / kf.get_n_splits()
avg_perm_null = perm_null_accumulator / kf.get_n_splits()

print("\nCross-Validated RMSE (hours):", round(avg_rmse_hours, 2))

stageB = pd.DataFrame({
    "Feature": X_model.columns,
    "Gain_Importance": avg_gain,
    "Permutation_Real": avg_perm_real, #real data
    "Permutation_Null": avg_perm_null, #shuffled target (noise)
    "Permutation_Delta": avg_perm_real - avg_perm_null #Real - Null
})

stageB_keep = stageB[stageB["Permutation_Delta"] > 0].copy()
stageB_keep = stageB_keep.sort_values("Permutation_Delta", ascending=False)

print("\nTop 20 Features — Stage B (Permutation Importance in HOURS, noise-adjusted):\n")
print(stageB_keep.head(20).to_string(index=False))


#######################################################################################################################################################
print("\n\nStage C starts Here\n")
###############################
########## STAGE C #########
###############################
# --- Safety checks ---
if "Eng. AH" in X.columns:
    X = X.drop(columns=["Eng. AH"], errors="ignore")

# Make sure everything is numeric
X = X.select_dtypes(include=[np.number]).copy()
X = X.fillna(0)

# Ensure y is numeric
y = y.astype(float)

print("RFECV input shapes:")
print("X:", X.shape)
print("y:", y.shape)

# --- CV setup ---
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# --- Model (keep it light for speed) ---
xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

# --- RMSE scorer (RFECV needs a "higher is better" scorer, so we use negative RMSE) ---
def rmse_hours(y_true_log, y_pred_log):
    y_true_h = np.exp(y_true_log)
    y_pred_h = np.exp(y_pred_log)
    return np.sqrt(mean_squared_error(y_true_h, y_pred_h))

rmse_scorer = make_scorer(rmse_hours, greater_is_better=False)

# --- RFECV ---
rfecv = RFECV(
    estimator=xgb_model,
    step=1,                 # remove 1 feature at a time (best for small feature sets like yours)
    cv=cv,
    scoring=rmse_scorer,
    min_features_to_select=max(10, int(0.25 * X.shape[1])),
    n_jobs=-1
)

rfecv.fit(X, y)

# --- Results ---
support_mask = rfecv.support_
selected_features = X.columns[support_mask].tolist()
ranking = rfecv.ranking_

print("\n\nRFECV finished")
print("Best number of features:", rfecv.n_features_)
print("Selected features:\n", selected_features)

# --- Create a clean results table ---
rfecv_results = pd.DataFrame({
    "Feature": X.columns,
    "Selected": support_mask,
    "Rank": ranking
}).sort_values(["Selected", "Rank"], ascending=[False, True])

print("\nTop selected features (ranked):")
print(rfecv_results[rfecv_results["Selected"] == True].to_string(index=False))

print("\nDropped features (ranked):")
print(rfecv_results[rfecv_results["Selected"] == False].head(30).to_string(index=False))


print("\nInitial cleaned dataset:")
print("Shape:", model_data.shape)   # <-- replace with your original dataset variable

print("\n============================================================")


# --- Build final dataset for modeling ---
X_final = X[selected_features].copy()

print("\nFinal modeling matrix:")
print("X_final shape:", X_final.shape)
print("y shape:", y.shape)

# # Optional: save for proof + sharing
# rfecv_results.to_excel("rfecv_feature_ranking.xlsx", index=False)
# print("\nSaved: rfecv_feature_ranking.xlsx")

# ============================================================
# Slide-ready summary (Feature count + RMSE before/after RFECV)
# ============================================================

# "How many did we start with?" (feature count used in modeling before RFECV)
start_features = X.shape[1]              # features going into RFECV
end_features   = X_final.shape[1]        # features kept by RFECV
rows_used      = X.shape[0]

# RMSE before RFECV = Stage B cross-validated RMSE in real hours
rmse_before = avg_rmse_hours  # already computed in Stage B

# RMSE after RFECV (use RFECV CV results)
# RFECV uses "higher is better"; with greater_is_better=False it stores NEGATIVE RMSE scores
rmse_curve = -rfecv.cv_results_["mean_test_score"]   # converts to positive RMSE (hours)
best_idx = int(np.argmin(rmse_curve))
rmse_after = float(rmse_curve[best_idx])

# Feature counts tested in RFECV correspond to: min_features_to_select ... start_features
min_feats = max(10, int(0.25 * start_features))
best_n_features_from_curve = min_feats + best_idx   # should match rfecv.n_features_

print("\n" + "="*62)
print("FEATURE OPTIMIZATION RESULTS (Stage C: RFECV)")
print("="*62)
print(f"Rows used for modeling: {rows_used}")
print(f"Cleaned dataset (incl. target) shape: {model_data.shape}")
print(f"Starting feature set (X) shape: {X.shape}")
print(f"Final feature set (X_final) shape: {X_final.shape}")

print("\nFeature reduction:")
print(f"  {start_features} → {end_features}  (dropped {start_features - end_features})")

print("\nCross-validated RMSE (hours):")
print(f"  Before RFECV (Stage B XGB, 5-fold): {rmse_before:.2f}")
print(f"  After  RFECV (best subset):         {rmse_after:.2f}")

print("\nSummary:")
print(f"Reduced dimensionality from {start_features} to {end_features} features")
print(f"while maintaining cross-validated RMSE ({rmse_before:.2f} -> {rmse_after:.2f} hours).")
print(f"Why 17? RFECV tested subsets and the lowest CV RMSE occurred at {best_n_features_from_curve} features.")
print("="*62 + "\n")

###############################################################################
####################### MODEL TRAINING EVALUATION #############################
##############################################################################

print("X_final:", X_final.shape)
print("y:", y.shape)


# 0) Freeze the modeling dataset (use shrunken feature set)
X_model = X_final.copy()
y_model = y.copy()   # this is LOG(hours) in your pipeline

print("X_model:", X_model.shape, "y_model:", y_model.shape)

# 1) Train / Val / Test split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_model, y_model, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# Helper: evaluate on LOG target and also report metrics in REAL HOURS (RMSE/MAE)
def eval_model(model, X, y_log):
    pred_log = model.predict(X)

    # Metrics on log scale (fine for R2)
    r2 = r2_score(y_log, pred_log)

    # Convert to real hours for MAE/RMSE (more interpretable)
    y_h = np.exp(y_log)
    pred_h = np.exp(pred_log)

    mae = mean_absolute_error(y_h, pred_h)
    rmse = np.sqrt(mean_squared_error(y_h, pred_h))

    return r2, mae, rmse

# 2) Baseline
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

rows = []
for split_name, Xs, ys in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
    r2, mae, rmse = eval_model(baseline, Xs, ys)
    rows.append({"model": "BaselineMean", "split": split_name, "R2": r2, "MAE_hours": mae, "RMSE_hours": rmse})

# 3) Test 5 regression models
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(random_state=42),
    "Lasso": Lasso(random_state=42, max_iter=10000),
    "RandomForest": RandomForestRegressor(random_state=42, n_estimators=300),
    "XGBoost": XGBRegressor(
        random_state=42, n_estimators=300, max_depth=4,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=-1
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    for split_name, Xs, ys in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        r2, mae, rmse = eval_model(model, Xs, ys)
        rows.append({"model": name, "split": split_name, "R2": r2, "MAE_hours": mae, "RMSE_hours": rmse})

results = pd.DataFrame(rows)
print("\nRESULTS (all models):")
print(results.sort_values(["split", "RMSE_hours"]).to_string(index=False))

# 4) Pick best model based on VAL RMSE (hours)
val_results = results[results["split"] == "val"].sort_values("RMSE_hours")
best_model_name = val_results.iloc[0]["model"]
print("\nBest model based on VAL RMSE:", best_model_name)

# 5) Hyperparameter tuning (ONLY for best model — usually XGBoost or RandomForest)
if best_model_name == "XGBoost":
    param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    }

    xgb = models["XGBoost"]
    search = RandomizedSearchCV(
        xgb, param_distributions=param_dist,
        n_iter=20, cv=5, random_state=42,
        scoring="neg_mean_squared_error", n_jobs=-1
    )
    search.fit(X_train, y_train)

    best_tuned = search.best_estimator_
    print("Best params:", search.best_params_)

    for split_name, Xs, ys in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        r2, mae, rmse = eval_model(best_tuned, Xs, ys)
        print(f"Tuned XGB {split_name}: R2={r2:.3f}, MAE={mae:.2f}h, RMSE={rmse:.2f}h")

elif best_model_name == "RandomForest":
    param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 8, 12, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    rf = models["RandomForest"]
    search = RandomizedSearchCV(
        rf, param_distributions=param_dist,
        n_iter=20, cv=5, random_state=42,
        scoring="neg_mean_squared_error", n_jobs=-1
    )
    search.fit(X_train, y_train)

    best_tuned = search.best_estimator_
    print("Best params:", search.best_params_)

    for split_name, Xs, ys in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        r2, mae, rmse = eval_model(best_tuned, Xs, ys)
        print(f"Tuned RF {split_name}: R2={r2:.3f}, MAE={mae:.2f}h, RMSE={rmse:.2f}h")

#Checking if any binary columns still exist
#print([c for c in model_data.columns if "Binary" in c])



"""
Finalize feature selection in three stages: Univariate Screening (A), Model-Based Importance with CV (B), and Recursive Feature Elimination with CV (C)
to keep only features that truly improve predictive performance on our log-transformed target.

This ensures we:
 - remove noise and redundancy,
 - avoid overfitting,
 - and keep a compact, high-signal feature set for XGBoost/LightGBM/RF.


What to do next (high level)
     - Stage A — Univariate Screening
         - Quickly rank features by Mutual Information, Spearman correlation, and shallow tree importance to identify obvious weak/strong candidates.

     - Stage B — Model-Based Importance with CV
         - Train a tree-based model (XGBRegressor) with K-Fold cross-validation, and compute both gain importance and permutation importance on validation folds.
         This gives a robust, model-agnostic view of which features matter.

    - Stage C — RFECV (Recursive Feature Elimination with CV)
         - Use RFECV with the same tree model to find the smallest feature set that preserves (or improves) cross-validated RMSE on the log target.

We’ve already cleaned, encoded, removed near-constant and redundant features, and handled high-cardinality categoricals.
Now we need evidence-based feature selection to reduce dimensionality and improve generalization.
Univariate = fast signal check; Model-based = captures interactions and non-linearities; RFECV = builds a minimal yet strong feature set.
"""








































            ################## OLD CODE BELOW #################



# # ##################################################################
# #     #1) Load the Excel dataset
# # ##################################################################

# # 1: Import Libraries
# from scipy.stats import skew
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas.api.types as ptypes
# import warnings
# import seaborn as sns
# from pandas.api.types import is_numeric_dtype
# from scipy.stats import kruskal
# import statsmodels.api as sm
# from sklearn.preprocessing import LabelEncoder


# warnings.filterwarnings("ignore") # Ignore python warnings
# pd.set_option('display.max_columns', None) # Settings to print all columns in the dataframe
# pd.set_option('display.max_rows', None) # Settings to print all rows in the dataframe

# # 2: Download Data
# data = pd.read_excel("Final_data_Ctech.xlsx")
# print("Shape:", data.shape) #Prints (rows,columns)

#3-Delete columns we dont need(I did this manually in Excel)

# # ##################################################################
# #     #2) Gather Intital Information
# # ##################################################################
# print("##########################################")
# print("##### Dataset Preliminary Informaion #####")
# print("##########################################" + "\n")

# print("Dataframe head(5):")
# print(data.head(5)) # Prints the first 5 rows
# print()

# print("Dataframe shape:")
# print("Number of columns:", str(data.shape[1])) # Prints (rows,columns)
# print("Number of rows:", str(data.shape[0]) + '\n') # Prints (rows,columns)

# print("Descriptive Statistics:")
# print(data.describe()) # Generate descriptive statistics
# print()

# print("Dataframe dtypes:")
# print(data.dtypes) # Return the dtypes in the DataFrame
# print()

# print("Duplicated Rows: " + str(data.duplicated().sum()) + ("\n")) # Return boolean Series denoting duplicate rows

# #print(data["CCN_Data Hub"].nunique())
# #print(data["CCN_Data Hub"].value_counts())

# # ##################################################################
# #     #3) Early Deletion(Columns That are Not Valuable)
# # ##################################################################

# data = data.drop(columns=[ #I will leave this step for you since you have the best understanding of which columns are needed
                           
      



# ])
# # #print(df_clean.columns.tolist())

# # ##################################################################

# # ##################################################################
# #     #4) Missing Percentage / Distribution type 
# # ##################################################################
# data["Eng. SH"] = pd.to_numeric(data["Eng. SH"], errors="coerce")
# data["Lab. SH"] = pd.to_numeric(data["Lab. SH"], errors="coerce")
# data["Eng. AH"] = pd.to_numeric(data["Eng. AH"], errors="coerce")
# data["Lab. AH"] = pd.to_numeric(data["Lab. AH"], errors="coerce")

# missing_count = data.isna().sum() #Finds the sum of N/A values for each column
# missing_percent = (data.isna().mean()*100).round(2) #Finds the percentage of N/A values for each column

# distribution = {} #Creates an empty dictionary to store the results for each column.
# for col in data.columns: #Loops through every column in the dataset.
#     col_series = data[col].dropna() #Removes any missing values

#     if col_series.empty:  #If the column has no data left after removing missing values, label it as "No data".
#         distribution[col] = "No data"
#     elif not is_numeric_dtype(data[col]):
#         distribution[col] = "Categorical" #If the column is not numeric, label it as Categorical.
#     else:
#         s = skew(col_series)#If the column is numeric, calculate its skewness using the skew() function.
#         if s >= 1:
#             distribution[col] = "Right skewed"  #If skewness is 1 or more, the data has a long tail on the right (a few large values).
#         elif s <= -1:
#             distribution[col] = "Left skewed" #If skewness is -1 or less, the data has a long tail on the left (a few small values).
#         elif abs(s) < 0.5:
#             distribution[col] = "normal" #If skewness is close to 0 (between -0.5 and 0.5), the data is fairly symmetrical.
#         else:
#             distribution[col] = "Slightly skewed"


# #Table for % of missing values
# summary = pd.DataFrame({
#         "Data Type" : data.dtypes.astype(str), #Type of data
#         "Missing Count": missing_count, #Total # of missing values per column
#         "Missing %": missing_percent,    #Percentage of missing values per column
#         "  Distribution": distribution
#         }).sort_values("Missing %", ascending=False) #Sorts the values in ascending order

# print("\nMISSING VALUES PER ATTRIBUTE + DISTRIBUTION")
# print(summary.to_string())

# # ########################################################################################################
# # #VISUALIZING THE DISTRIBUTION

# # # Loop through numeric columns
# # for col in data.select_dtypes(include=['int64', 'float64']).columns:
# #     plt.figure(figsize=(6,4))
# #     sns.histplot(data[col].dropna(), kde=True, bins=30)
# #     plt.title(f"Distribution of {col}")
# #     plt.xlabel(col)
# #     plt.ylabel("Frequency")
# #     plt.show()

# # # Loop through categorical columns
# # for col in data.select_dtypes(include=['object', 'category']).columns:
# #     plt.figure(figsize=(8,4))
# #     sns.countplot(x=data[col])
# #     plt.title(f"Count Plot of {col}")
# #     plt.xticks(rotation=45)
# #     plt.show()



# # ##################################################################
# #     #5) Data Visualization
# # ##################################################################

# #Plot the frequency of each investigation type
# types = [
#     "1 - Full Investigation",
#     "2 - Full Investigation + Alternate Construction",
#     "3 - Alternate Construction",
#     "4 - Administrative No Test anticipated (revisions requiring Engineering Review)",
#     "5 - Administrative CB review"
# ]

# counts = {t: (data["type_of_investigation"] == t).sum() for t in types}
# print(counts)
# plt.figure(figsize=(6, 8))
# ax = pd.Series(counts).plot(kind="bar", color="skyblue")
# total = sum(counts.values())
# print("Total:", total)
# labels = [f"{c} ({c/total*100:.1f}%)" for c in counts.values()]
# ax.bar_label(ax.containers[0], labels=labels)

# plt.xlabel("Type of Investigation")
# plt.ylabel("Count")
# plt.title("Frequency of Each Investigation Type")
# plt.title('Average Total Actual Hours by Investigation Type')
# plt.legend(['Investigation Type'])
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()

# # #Calcuates the average total lab actual hours for each investigation type
# avg_lab = (
#     data.loc[data["type_of_investigation"].isin(types)]
#         .groupby("type_of_investigation")["Lab. AH"]
#         .mean()
#         .reindex(types)
# )

# median_lab = (
#     data.loc[data["type_of_investigation"].isin(types)]
#         .groupby("type_of_investigation")["Lab. AH"]
#         .median()
#         .reindex(types)
# )

# summary_lab = pd.DataFrame({"Mean": avg_lab, "Median": median_lab})

# ax = summary_lab.plot(kind="bar", figsize=(10, 6))
# plt.xlabel("Type of Investigation")
# plt.ylabel("Hours")
# plt.title("Mean vs Median Lab Actual Hours by Investigation Type")
# plt.legend(["Mean", "Median"])
# plt.xticks(rotation=30, ha="right")

# # add labels on top of bars
# for container in ax.containers:
#     ax.bar_label(container, fmt="%.1f")

# plt.tight_layout()
# plt.show()

# ###Calcuates the average total actual eng hours for each investigation type
# avg_eng = (
#     data.loc[data["type_of_investigation"].isin(types)]
#         .groupby("type_of_investigation")["Eng. AH"]
#         .mean()
#         .reindex(types)
# )

# median_eng = (
#     data.loc[data["type_of_investigation"].isin(types)]
#         .groupby("type_of_investigation")["Eng. AH"]
#         .median()
#         .reindex(types)
# )

# summary_eng = pd.DataFrame({"Mean": avg_eng, "Median": median_eng})

# ax = summary_eng.plot(kind="bar", figsize=(10, 6))
# plt.xlabel("Type of Investigation")
# plt.ylabel("Hours")
# plt.title("Mean vs Median Eng Actual Hours by Investigation Type")
# plt.legend(["Mean", "Median"])
# plt.xticks(rotation=30, ha="right")

# # add labels on top of bars
# for container in ax.containers:
#     ax.bar_label(container, fmt="%.1f")

# plt.tight_layout()
# plt.show()

# # ##################################################################
# # #6) correlation between the variables and the targets
# # ##################################################################

# print("############################### Lab Actual Hours Correlation to Other Attributes #################################")
# corr = data.corr(numeric_only=True)
# print(corr["Lab. AH"].sort_values(ascending=False))
# print("############################### Eng Actual Hours Correlation to Other Attributes #################################")
# print(corr["Eng. AH"].sort_values(ascending=False))



# # corr = data.corr(numeric_only=True)
# # drop_cols = [col for col in corr.columns if "hrs" in col.lower()]
# # lab_corr = corr ["Lab. AH"].drop(drop_cols).sort_values(ascending=False)
# # eng_corr = corr ["Eng. AH"].drop(drop_cols).sort_values(ascending=False)
# # comparison = pd.DataFrame({
# #     "Lab. AH": lab_corr,
# #     "Eng. AH": eng_corr,
# # })
# # print(comparison)



# #I am still trying to understand hoe outliers work; however, from a quick glance it seems that most projects have normal hours,
# #but there are some with extremely high hours (outliers). Some rows also have negative hour values, so I am assuming the data
# #needs some cleaning up before modeling. 


# #^^^^^^^^^^

# # ##################################################################
# # #7) Outliers to target
# # ##################################################################
# outlier_cols = ["Eng. SH", "Lab. SH", "Eng. AH", "Lab. AH"]

# # (optional but useful) totals
# data["Total SH"] = data["Eng. SH"] + data["Lab. SH"]
# data["Total AH"] = data["Eng. AH"] + data["Lab. AH"]
# outlier_cols += ["Total SH", "Total AH"]

# # Negative / invalid hours check
# neg = data[(data[outlier_cols] < 0).any(axis=1)]
# print("\nNEGATIVE HOURS ROWS:", neg.shape[0])
# if neg.shape[0] > 0:
#     print(neg[outlier_cols].head(10))

# # IQR outlier check (handles zero-heavy columns)
# print("\nIQR OUTLIER CHECK (1.5 * IQR):")
# for col in outlier_cols:

#     # Use only non-zero values for IQR so Lab columns don't get bounds [0,0]
#     temp = data.loc[data[col] > 0, col].dropna()

#     # If column is all zeros / empty, skip
#     if temp.empty:
#         print(f"\n--- {col} ---")
#         print("Skipped (all zeros or no data)")
#         continue

#     Q1 = temp.quantile(0.25)
#     Q3 = temp.quantile(0.75)
#     IQR = Q3 - Q1

#     # If IQR is still 0 (too many identical values), skip
#     if IQR == 0:
#         print(f"\n--- {col} ---")
#         print("Skipped (IQR = 0, values too concentrated)")
#         continue

#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR

#     outliers = data[(data[col] > 0) & ((data[col] < lower) | (data[col] > upper))]

#     print(f"\n--- {col} ---")
#     print(f"Bounds: [{lower:.2f}, {upper:.2f}]")
#     print(f"Outliers Found: {outliers.shape[0]}")
#     print("Top 5 outlier values:")
#     print(outliers[col].sort_values(ascending=False).head(5))













#####################DUPLICATES CODE###########################
# # Excel row number (assumes row 1 in Excel is headers)
# # If your sheet has extra header rows, adjust +2 to +3/+4 etc.
# data.insert(0, "excel_row", data.index + 2)

# # --- find exact duplicate rows across ALL columns (excluding excel_row) ---
# subset_cols = data.columns.drop("excel_row")

# dups = data[data.duplicated(subset=subset_cols, keep=False)].copy()

# # assign a group id so duplicate copies sit together
# dups["dup_group"] = dups.groupby(list(subset_cols), dropna=False).ngroup() + 1

# # sort so pairs are next to each other
# dups = dups.sort_values(["dup_group", "excel_row"])

# # save report for Excel verification
# dups.to_excel("DUPLICATES_REPORT_WITH_EXCEL_ROWS.xlsx", index=False)

# print("Saved: DUPLICATES_REPORT_WITH_EXCEL_ROWS.xlsx")
# print("Duplicate groups:", dups["dup_group"].nunique())
# print("Duplicate rows total:", len(dups))







#STAGE A combination 

# #PART 4(EXTRA STEP): COMPARE ALL 3

# print("\n\n==================================================")
# print("Top 20 Features (MI + Spearman + Tree aligned)")
# print("====================================================\n")

# # Set Feature as index for safe alignment
# mi_tmp = mi_df.set_index("Feature")
# sp_tmp = spearman_df.set_index("Feature")
# tree_tmp = tree_importance.set_index("Feature")

# # Combine all into one table
# comparison_all = pd.concat(
#     [
#         mi_tmp["MI_Score"],
#         sp_tmp["Spearman"],
#         sp_tmp["Abs_Spearman"],
#         tree_tmp["Tree_Importance"]
#     ],
#     axis=1
# )

# # Optional normalization
# comparison_all["MI_Normalized"] = (
#     comparison_all["MI_Score"] /
#     comparison_all["MI_Score"].max()
# )

# comparison_all["Tree_Normalized"] = (
#     comparison_all["Tree_Importance"] /
#     (comparison_all["Tree_Importance"].max() if comparison_all["Tree_Importance"].max() > 0 else 1)
# )

# # Sort by MI (or change if you prefer)
# comparison_all = comparison_all.sort_values(
#     by=["MI_Score", "Abs_Spearman", "Tree_Importance"],
#     ascending=False
# )

# # Reset index so Feature becomes a column again
# comparison_all = comparison_all.reset_index()

# # Clean print
# print(comparison_all.head(20).to_string(index=False))