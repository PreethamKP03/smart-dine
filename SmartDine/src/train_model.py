"""
SmartDine — Model Training & Comparison Module
src/train_model.py

Responsibilities:
1. Load train and test splits (train_ml.csv, test_ml.csv).
2. Compare 4 candidate supervised regression models:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
3. Evaluate metrics: MAE, MSE, RMSE, R²
4. Empirically select champion model and serialize to models/smartdine_best_model.joblib
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train_ml.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test_ml.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_and_compare_models():
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    feature_columns = [
        "group_size", "cuisine_match", "budget_match", "dietary_compatibility",
        "restaurant_type_match", "location_match", "booking_match", "online_order_match",
        "group_agreement", "rating_score", "popularity_score", "cost_for_two",
        "votes", "online_order", "book_table"
    ]
    target_column = "suitability_score"

    X_train = train_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_columns].median())
    y_train = train_df[target_column]
    X_test = test_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_columns].median())
    y_test = test_df[target_column]

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=60, max_depth=5, learning_rate=0.1, random_state=42)
    }

    results = []
    trained_models = {}

    print("\nTraining and evaluating models:")
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - t0
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "Training_Time_Seconds": duration
        })
        trained_models[name] = model
        print(f" - {name:20s} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f} | Time: {duration:.2f}s")

    comparison_df = pd.DataFrame(results)
    comparison_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)

    # Select best model based on R² and MAE
    best_row = comparison_df.sort_values(by=["R2", "MAE"], ascending=[False, True]).iloc[0]
    best_name = best_row["Model"]
    best_model = trained_models[best_name]

    print("\n" + "=" * 60)
    print(f"CHAMPION MODEL SELECTED: {best_name} (R² = {best_row['R2']:.4f})")
    print("=" * 60)

    # Serialize Best Model
    model_save_path = os.path.join(MODEL_DIR, "smartdine_best_model.joblib")
    joblib.dump(best_model, model_save_path)

    # Save feature columns
    feat_save_path = os.path.join(MODEL_DIR, "feature_columns.txt")
    with open(feat_save_path, "w") as f:
        for col in feature_columns:
            f.write(f"{col}\n")

    return best_name, comparison_df


if __name__ == "__main__":
    train_and_compare_models()
