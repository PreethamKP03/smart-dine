"""
SmartDine — Model Evaluation Module
src/evaluation.py

Responsibilities:
1. Load test dataset and champion trained model.
2. Calculate comprehensive regression metrics (MAE, MSE, RMSE, R²).
3. Generate diagnostic plots:
   - Actual vs. Predicted scatter plot
   - Residual distribution histogram
   - Feature coefficients importance plot
4. Save metrics to outputs/model_evaluation_metrics.csv
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test_ml.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "smartdine_best_model.joblib")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_columns.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_system():
    print("Evaluating SmartDine trained model...")
    test_df = pd.read_csv(TEST_PATH)
    with open(FEATURE_PATH, "r") as f:
        feature_columns = [l.strip() for l in f if l.strip()]

    X_test = test_df[feature_columns]
    y_test = test_df["suitability_score"]

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)
    print(f"MAE  : {mae:.5f}")
    print(f"MSE  : {mse:.5f}")
    print(f"RMSE : {rmse:.5f}")
    print(f"R²   : {r2:.5f}")

    metrics_df = pd.DataFrame([
        {"Metric": "MAE", "Value": mae},
        {"Metric": "MSE", "Value": mse},
        {"Metric": "RMSE", "Value": rmse},
        {"Metric": "R2", "Value": r2}
    ])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "model_evaluation_metrics.csv"), index=False)

    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test[:2000], y_pred[:2000], alpha=0.3, color="#3b82f6", s=15)
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal Fit (y = x)")
    plt.xlabel("Actual Ground Truth Score")
    plt.ylabel("Predicted Compatibility Score")
    plt.title("SmartDine: Actual vs. Predicted Group Compatibility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=300)
    plt.close()

    # 2. Residual Distribution Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True, color="#10b981", bins=40)
    plt.xlabel("Prediction Error (Residual)")
    plt.title("Residual Distribution (Error Centering at 0)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "residual_distribution.png"), dpi=300)
    plt.close()

    # 3. Feature Coefficients (if Linear Model)
    if hasattr(model, "coef_"):
        coef_df = pd.DataFrame({
            "Feature": feature_columns,
            "Coefficient": model.coef_
        }).sort_values(by="Coefficient", ascending=False)
        coef_df.to_csv(os.path.join(OUTPUT_DIR, "feature_coefficients.csv"), index=False)

        plt.figure(figsize=(9, 6))
        sns.barplot(data=coef_df, x="Coefficient", y="Feature", palette="viridis")
        plt.title("Feature Coefficients Impact on Group Compatibility")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "feature_coefficients.png"), dpi=300)
        plt.close()

    print("Diagnostic plots generated and saved in outputs/ directory.")


if __name__ == "__main__":
    evaluate_system()
