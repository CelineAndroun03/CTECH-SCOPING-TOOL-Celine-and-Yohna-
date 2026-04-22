import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

df = pd.read_csv("../artifacts/lab_cleaned_final.csv")

features = [
    "standard_count",
    "total_CB_count",
    "total_test_count",

    "62368-3",
    "IT Informational Test Report",
    "CB",
    "Standard Upgrades"
]


target = "Lab. AH"

X = df[features]
y = df[target]

model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = -cross_val_score(
    model, X, y, cv=kf, scoring="neg_root_mean_squared_error"
)

mae_scores = -cross_val_score(
    model, X, y, cv=kf, scoring="neg_mean_absolute_error"
)

r2_scores = cross_val_score(
    model, X, y, cv=kf, scoring="r2"
)

print("\nMODEL PERFORMANCE")
print(f"RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
print(f"MAE:  {mae_scores.mean():.4f}")
print(f"R2:   {r2_scores.mean():.4f}")
