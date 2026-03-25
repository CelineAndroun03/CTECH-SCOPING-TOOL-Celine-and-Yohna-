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
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# ==================================================
# 1. Import Data
# ==================================================
data = pd.read_excel("1. Data.xlsx")
data = data.drop_duplicates()

# ==================================================
# 2. Model
# ==================================================

df = data.copy()

X = df.drop(columns=["Lab. AH"])
y = df["Lab. AH"].astype(float)

bins = pd.qcut(y, q=10, duplicates='drop').astype("category").cat.codes

X_temp, X_test, y_temp, y_test, bins_temp, bins_test = train_test_split(X, y, bins, test_size=0.15, random_state=42, stratify=bins)
val_ratio = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=bins_temp)

print("Shapes:")
print(" Train:", X_train.shape)
print(" Valid:", X_val.shape)
print(" Test :", X_test.shape)


model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    verbosity=0
)

model.fit(X_train, y_train)

def to_hours(x):
    return np.maximum(0.0, np.expm1(x))

def evaluate(X, y):
    pred = model.predict(X)
    y_h = to_hours(y)
    p_h = to_hours(pred)
    mae = mean_absolute_error(y_h, p_h)
    rmse = np.sqrt(mean_squared_error(y_h, p_h))
    r2 = r2_score(y_h, p_h)
    return mae, rmse, r2

train_mae, train_rmse, train_r2 = evaluate(X_train, y_train)
val_mae,   val_rmse,   val_r2   = evaluate(X_val,   y_val)
test_mae,  test_rmse,  test_r2  = evaluate(X_test,  y_test)

print("\n================ RESULTADOS XGBRegressor ================")
print(f"Train -> MAE: {train_mae:.4f} | RMSE: {train_rmse:.4f} | R2: {train_r2:.4f}")
print(f"Valid -> MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | R2: {val_r2:.4f}")
print(f"Test  -> MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | R2: {test_r2:.4f}")
print("==========================================================")


# ==========================================
# XGBRegressor - Hyperparameter Tuning (Random Search + Early Stopping)
# ==========================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---------- Dados e split (igual ao teu) ----------
df = data.copy()

X = df.drop(columns=["Lab. AH"])
y = df["Lab. AH"].astype(float)

# Estratificação por bins do target (log1p)
bins = pd.qcut(y, q=10, duplicates='drop').astype("category").cat.codes

# 70/15/15
X_temp, X_test, y_temp, y_test, bins_temp, bins_test = train_test_split(
    X, y, bins, test_size=0.15, random_state=42, stratify=bins
)
val_ratio = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=bins_temp
)

print("Shapes:")
print(" Train:", X_train.shape)
print(" Valid:", X_val.shape)
print(" Test :", X_test.shape)

# ---------- Helpers ----------
def to_hours(x):
    # Inverso do log1p + clip para evitar negativos
    return np.maximum(0.0, np.expm1(x))

def eval_metrics(model, Xd, yd):
    pred_log = model.predict(Xd)
    y_h = to_hours(yd)
    p_h = to_hours(pred_log)
    mae = mean_absolute_error(y_h, p_h)
    rmse = np.sqrt(mean_squared_error(y_h, p_h))
    r2 = r2_score(y_h, p_h)
    return mae, rmse, r2

# ---------- Espaço de procura (focado em reduzir overfitting) ----------
RANDOM_STATE = 42
rs = np.random.RandomState(RANDOM_STATE)
N_TRIALS = 50

depth_choices = [3, 4, 5, 6, 7, 8, 9, 10]
mcw_choices   = [1, 2, 3, 5, 7, 10]
sub_choices   = [0.6, 0.7, 0.8, 0.9, 1.0]
col_choices   = [0.6, 0.7, 0.8, 0.9, 1.0]
gamma_choices = [0.0, 0.1, 0.3, 0.7, 1.0, 2.0]
eta_choices   = [0.01, 0.03, 0.05, 0.07, 0.1]

def sample_loguniform(low, high):
    # amostragem log-uniforme (útil para regularização)
    return float(np.exp(rs.uniform(np.log(low), np.log(high))))

results = []  # guarda resultados de cada trial

for t in range(1, N_TRIALS + 1):
    params = {
        "max_depth":         int(rs.choice(depth_choices)),
        "min_child_weight":  int(rs.choice(mcw_choices)),
        "subsample":         float(rs.choice(sub_choices)),
        "colsample_bytree":  float(rs.choice(col_choices)),
        "gamma":             float(rs.choice(gamma_choices)),
        "learning_rate":     float(rs.choice(eta_choices)),
        # regularização L1/L2 com amostragem log-uniforme (favorece valores pequenos mas não nulos)
        "reg_alpha":         sample_loguniform(1e-4, 1.0),   # L1
        "reg_lambda":        sample_loguniform(1e-3, 10.0),  # L2
    }

    model = XGBRegressor(
        n_estimators=2000,          # alto para permitir early stopping
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
        **params
    )

    # Early stopping no validation set (monitoriza RMSE no espaço do y_log)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=50,
        verbose=False
    )

    # Avaliação em horas
    tr_mae, tr_rmse, tr_r2 = eval_metrics(model, X_train, y_train)
    va_mae, va_rmse, va_r2 = eval_metrics(model, X_val, y_val)
    te_mae, te_rmse, te_r2 = eval_metrics(model, X_test, y_test)

    results.append({
        "trial": t,
        "params": params,
        "best_iteration": getattr(model, "best_iteration", None),
        "train_MAE": tr_mae, "train_RMSE": tr_rmse, "train_R2": tr_r2,
        "val_MAE":   va_mae, "val_RMSE":   va_rmse, "val_R2":   va_r2,
        "test_MAE":  te_mae, "test_RMSE":  te_rmse, "test_R2":  te_r2,
        "model": model
    })

    if t % 10 == 0 or t == N_TRIALS:
        print(f"[{t}/{N_TRIALS}] val_RMSE (melhor até agora): "
              f"{min(r['val_RMSE'] for r in results):.4f}")

# ---------- Selecionar o melhor por RMSE em validação ----------
results_sorted = sorted(results, key=lambda d: d["val_RMSE"])
best = results_sorted[0]
print("\n===== TOP-5 combinações por RMSE (Validation) =====")
for r in results_sorted[:5]:
    print(f"Trial {r['trial']:>3} | val_RMSE={r['val_RMSE']:.4f} | "
          f"train_RMSE={r['train_RMSE']:.4f} | test_RMSE={r['test_RMSE']:.4f} | "
          f"best_iter={r['best_iteration']} | params={r['params']}")

# ---------- Métricas finais do melhor modelo ----------
best_model = best["model"]
train_mae, train_rmse, train_r2 = best["train_MAE"], best["train_RMSE"], best["train_R2"]
val_mae,   val_rmse,   val_r2   = best["val_MAE"],   best["val_RMSE"],   best["val_R2"]
test_mae,  test_rmse,  test_r2  = best["test_MAE"],  best["test_RMSE"],  best["test_R2"]

print("\n================ MELHOR MODELO (depois de tuning) ================")
print(f"Best iteration (early stopping): {best['best_iteration']}")
print(f"Params: {best['params']}")
print(f"Train -> MAE: {train_mae:.4f} | RMSE: {train_rmse:.4f} | R2: {train_r2:.4f}")
print(f"Valid -> MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | R2: {val_r2:.4f}")
print(f"Test  -> MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | R2: {test_r2:.4f}")
print("==================================================================")
