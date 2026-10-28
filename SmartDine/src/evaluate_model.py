# src/evaluate_model.py

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PATHS
# ============================================================

TEST_PATH = "data/processed/test_ml.csv"
MODEL_PATH = "models/smartdine_best_model.joblib"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("Loading test dataset...")

test_df = pd.read_csv(TEST_PATH)

print(f"Test dataset shape: {test_df.shape}")


# ============================================================
# 3. LOAD FEATURE LIST
# ============================================================

FEATURE_PATH = "models/feature_columns.txt"

with open(FEATURE_PATH, "r") as f:
    feature_columns = [line.strip() for line in f if line.strip()]

print("\nFeatures used by the model:")
for feature in feature_columns:
    print(f" - {feature}")


# ============================================================
# 4. PREPARE X AND Y
# ============================================================

X_test = test_df[feature_columns]
y_test = test_df["suitability_score"]


# ============================================================
# 5. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

print(f"Model: {model}")


# ============================================================
# 6. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)


# ============================================================
# 7. CALCULATE METRICS
# ============================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")


# ============================================================
# 8. ACTUAL VS PREDICTED PLOT
# ============================================================

print("\nCreating Actual vs Predicted plot...")

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.15,
    s=10
)

# Perfect prediction line
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Suitability Score")
plt.ylabel("Predicted Suitability Score")
plt.title("Actual vs Predicted Suitability Score")

plt.tight_layout()

actual_predicted_path = os.path.join(
    OUTPUT_DIR,
    "actual_vs_predicted.png"
)

plt.savefig(actual_predicted_path, dpi=300)
plt.close()

print(f"Saved: {actual_predicted_path}")


# ============================================================
# 9. RESIDUAL CALCULATION
# ============================================================

residuals = y_test - y_pred


# ============================================================
# 10. RESIDUAL DISTRIBUTION
# ============================================================

print("Creating residual distribution plot...")

plt.figure(figsize=(8, 6))

plt.hist(
    residuals,
    bins=50,
    edgecolor="black"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")

plt.tight_layout()

residual_path = os.path.join(
    OUTPUT_DIR,
    "residual_distribution.png"
)

plt.savefig(residual_path, dpi=300)
plt.close()

print(f"Saved: {residual_path}")


# ============================================================
# 11. FEATURE COEFFICIENTS
# ============================================================

print("Creating feature coefficient plot...")

coefficients = model.coef_

coef_df = pd.DataFrame({
    "Feature": feature_columns,
    "Coefficient": coefficients
})

# Sort by coefficient value
coef_df = coef_df.sort_values(
    "Coefficient",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    coef_df["Feature"],
    coef_df["Coefficient"]
)

plt.xlabel("Linear Regression Coefficient")
plt.ylabel("Feature")
plt.title("Feature Contributions to Suitability Prediction")

plt.tight_layout()

coefficient_path = os.path.join(
    OUTPUT_DIR,
    "feature_coefficients.png"
)

plt.savefig(coefficient_path, dpi=300)
plt.close()

print(f"Saved: {coefficient_path}")


# ============================================================
# 12. SAVE COEFFICIENT TABLE
# ============================================================

coefficient_csv_path = os.path.join(
    OUTPUT_DIR,
    "feature_coefficients.csv"
)

coef_df.to_csv(
    coefficient_csv_path,
    index=False
)

print(f"Saved: {coefficient_csv_path}")


# ============================================================
# 13. SAVE EVALUATION METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2"
    ],
    "Value": [
        mae,
        rmse,
        r2
    ]
})

metrics_path = os.path.join(
    OUTPUT_DIR,
    "model_evaluation_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)

print(f"Saved: {metrics_path}")


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated files:")

print("1. outputs/actual_vs_predicted.png")
print("2. outputs/residual_distribution.png")
print("3. outputs/feature_coefficients.png")
print("4. outputs/feature_coefficients.csv")
print("5. outputs/model_evaluation_metrics.csv")

print("\nBest Model: Linear Regression")
print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")