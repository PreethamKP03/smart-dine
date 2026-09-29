import pandas as pd
import numpy as np
import os


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "restaurants_cleaned.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "restaurant_features.csv"
)


# ============================================================
# 2. LOAD CLEANED RESTAURANT DATASET
# ============================================================

print("Loading cleaned restaurant dataset...")

restaurants = pd.read_csv(INPUT_PATH)

print("Dataset shape:", restaurants.shape)


# ============================================================
# 3. CREATE RESTAURANT-LEVEL FEATURES
# ============================================================

print("\nCreating restaurant-level features...")


# Popularity transformation
restaurants["log_votes"] = np.log1p(restaurants["votes"])


# Number of cuisines offered by each restaurant
def count_cuisines(value):
    if pd.isna(value):
        return 0

    cuisines = [
        cuisine.strip()
        for cuisine in str(value).split(",")
        if cuisine.strip()
    ]

    return len(cuisines)


restaurants["cuisine_count"] = (
    restaurants["cuisines"].apply(count_cuisines)
)


# Availability indicators
restaurants["has_rating"] = (
    restaurants["rating"].notna().astype(int)
)

restaurants["has_cost"] = (
    restaurants["cost_for_two"].notna().astype(int)
)

restaurants["has_booking"] = (
    restaurants["book_table"].astype(int)
)

restaurants["has_online_order"] = (
    restaurants["online_order"].astype(int)
)


# ============================================================
# 4. SELECT FEATURES
# ============================================================

feature_columns = [
    "restaurant_id",
    "name",
    "location",
    "cuisines",
    "rest_type",
    "listed_in_type",
    "rating",
    "votes",
    "log_votes",
    "cost_for_two",
    "online_order",
    "book_table",
    "cuisine_count",
    "has_rating",
    "has_cost",
    "has_booking",
    "has_online_order"
]

restaurant_features = restaurants[feature_columns].copy()


# ============================================================
# 5. SAVE RESTAURANT FEATURE DATASET
# ============================================================

restaurant_features.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 6. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("RESTAURANT FEATURE ENGINEERING COMPLETED")
print("=" * 70)

print("Input shape:", restaurants.shape)
print("Output shape:", restaurant_features.shape)

print("\nFeature columns:")
print(restaurant_features.columns.tolist())

print("\nMissing values:")
print(
    restaurant_features
    .isna()
    .sum()
    .sort_values(ascending=False)
)

print("\nOutput file:")
print(OUTPUT_PATH)