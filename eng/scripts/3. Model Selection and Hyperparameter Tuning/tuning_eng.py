###############################################################################
####################### MODEL TRAINING EVALUATION #############################
##############################################################################

"""
Divide the clean dataset into 3 parts:
70% -> Training (Try 5 different models and see which one learns from the 70% training data, we are just learning patterns within our data in this step)
15% -> Validation (We test each model on the 15% validation data, calculating RMSE,MAE,R^2, trying different hyperparameters,using CV or regular validation) 
and Whichever model predicts best we choose. 
15% -> Final test (Take the best final tuned model, test it on the untouched 15% test data and this tells us if we deploy this in production, this is how accurate
our results would be)
"""

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

