import pandas as pd
import numpy as np
import os
import time
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "train_ml.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "test_ml.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading ML datasets...")

train_df = pd.read_csv(TRAIN_PATH)

test_df = pd.read_csv(TEST_PATH)

print("Training shape:", train_df.shape)

print("Testing shape:", test_df.shape)


# ============================================================
# 3. DEFINE FEATURES
# ============================================================

feature_columns = [

    "group_size",

    "cuisine_match",

    "budget_match",

    "dietary_compatibility",

    "restaurant_type_match",

    "location_match",

    "booking_match",

    "online_order_match",

    "group_agreement",

    "rating_score",

    "popularity_score",

    "cost_for_two",

    "votes",

    "online_order",

    "book_table"
]

target_column = "suitability_score"


X_train = train_df[
    feature_columns
].copy()

y_train = train_df[
    target_column
].copy()

X_test = test_df[
    feature_columns
].copy()

y_test = test_df[
    target_column
].copy()


# ============================================================
# 4. HANDLE NUMERICAL VALUES
# ============================================================

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)


X_train = X_train.fillna(
    X_train.median()
)

X_test = X_test.fillna(
    X_train.median()
)


# ============================================================
# 5. DEFINE MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            max_depth=12,
            min_samples_leaf=20,
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=20,
            random_state=42
        )
}


# ============================================================
# 6. TRAIN AND EVALUATE
# ============================================================

results = []

trained_models = {}


for model_name, model in models.items():

    print("\n" + "=" * 70)

    print(
        f"Training {model_name}..."
    )

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.time() - start_time
    )


    print("Generating predictions...")

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )


    results.append({

        "Model": model_name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2,

        "Training_Time_Seconds":
            training_time
    })


    trained_models[
        model_name
    ] = model


    print(
        f"MAE  : {mae:.6f}"
    )

    print(
        f"RMSE : {rmse:.6f}"
    )

    print(
        f"R²   : {r2:.6f}"
    )

    print(
        f"Time : {training_time:.2f} seconds"
    )


# ============================================================
# 7. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="RMSE"
)

print("\n" + "=" * 70)

print("MODEL COMPARISON")

print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# 9. SELECT BEST MODEL
# ============================================================

best_model_name = (
    results_df
    .iloc[0]["Model"]
)

best_model = trained_models[
    best_model_name
]


print("\n" + "=" * 70)

print(
    "BEST MODEL:",
    best_model_name
)

print("=" * 70)


# ============================================================
# 10. SAVE BEST MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "smartdine_best_model.joblib"
)

joblib.dump(
    best_model,
    model_path
)


# ============================================================
# 11. SAVE FEATURE LIST
# ============================================================

feature_path = os.path.join(
    MODEL_DIR,
    "feature_columns.txt"
)

with open(
    feature_path,
    "w"
) as file:

    for feature in feature_columns:

        file.write(
            feature + "\n"
        )


# ============================================================
# 12. FINAL OUTPUT
# ============================================================

print("\nSaved model:")
print(model_path)

print("\nSaved comparison:")
print(results_path)

print("\nSaved feature list:")
print(feature_path)

print("\nTraining completed successfully.")