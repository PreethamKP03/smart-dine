import pandas as pd
import numpy as np
import os


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "group_restaurant_features.csv"
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


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading group × restaurant dataset...")

df = pd.read_csv(INPUT_PATH)

print("Original shape:", df.shape)


# ============================================================
# 3. HANDLE MISSING RESTAURANT VALUES
# ============================================================

print("\nHandling missing restaurant attributes...")

# Rating
rating_median = df["rating"].median()

df["rating"] = (
    df["rating"]
    .fillna(rating_median)
)

# Cost
cost_median = df["cost_for_two"].median()

df["cost_for_two"] = (
    df["cost_for_two"]
    .fillna(cost_median)
)


# ============================================================
# 4. RECALCULATE RATING SCORE
# ============================================================

df["rating_score"] = (
    df["rating"] / 5.0
)


# ============================================================
# 5. RECALCULATE POPULARITY SCORE
# ============================================================

max_log_votes = df["popularity_score"].max()

if max_log_votes > 0:

    df["popularity_score"] = (
        df["popularity_score"]
        .clip(0, 1)
    )


# ============================================================
# 6. CORRECTED SYNTHETIC SUITABILITY TARGET
# ============================================================
#
# IMPORTANT:
#
# This is a synthetic benchmark target.
# It represents our defined suitability framework.
# It is NOT a real human rating.
#
# Group agreement is explicitly included here.
# ============================================================

print("\nCreating corrected suitability target...")

df["suitability_score"] = (

      0.20 * df["cuisine_match"]

    + 0.20 * df["budget_match"]

    + 0.15 * df["dietary_compatibility"]

    + 0.10 * df["restaurant_type_match"]

    + 0.10 * df["location_match"]

    + 0.10 * df["group_agreement"]

    + 0.05 * df["rating_score"]

    + 0.03 * df["booking_match"]

    + 0.03 * df["online_order_match"]

    + 0.04 * df["popularity_score"]
)


# Add small controlled noise so the target is not
# a perfectly deterministic weighted sum.

np.random.seed(42)

noise = np.random.normal(
    loc=0.0,
    scale=0.03,
    size=len(df)
)

df["suitability_score"] = (
    df["suitability_score"] + noise
)

df["suitability_score"] = (
    df["suitability_score"]
    .clip(0, 1)
)


# ============================================================
# 7. SELECT ML FEATURES
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


X = df[feature_columns].copy()

y = df[target_column].copy()


# ============================================================
# 8. CHECK FEATURES
# ============================================================

print("\nML feature columns:")

for column in feature_columns:
    print(" -", column)


print("\nFeature missing values:")

print(
    X.isna()
    .sum()
)


print("\nTarget statistics:")

print(
    y.describe()
)


# ============================================================
# 9. GROUP-WISE TRAIN / TEST SPLIT
# ============================================================
#
# IMPORTANT:
# We split by GROUP, not by individual rows.
#
# This prevents the same group from appearing in both
# training and testing data.
# ============================================================

print("\nCreating group-wise train/test split...")

unique_groups = (
    df["group_id"]
    .unique()
)

np.random.seed(42)

np.random.shuffle(
    unique_groups
)


split_index = int(
    len(unique_groups) * 0.80
)

train_groups = unique_groups[
    :split_index
]

test_groups = unique_groups[
    split_index:
]


train_mask = (
    df["group_id"]
    .isin(train_groups)
)

test_mask = (
    df["group_id"]
    .isin(test_groups)
)


train_df = df.loc[
    train_mask
].copy()

test_df = df.loc[
    test_mask
].copy()


# ============================================================
# 10. SAVE TRAINING DATA
# ============================================================

train_output = train_df[
    ["group_id", "restaurant_id"]
    + feature_columns
    + [target_column]
].copy()


test_output = test_df[
    ["group_id", "restaurant_id"]
    + feature_columns
    + [target_column]
].copy()


train_output.to_csv(
    TRAIN_PATH,
    index=False
)

test_output.to_csv(
    TEST_PATH,
    index=False
)


# ============================================================
# 11. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("ML DATASET PREPARATION COMPLETED")
print("=" * 70)

print(
    "Total rows:",
    len(df)
)

print(
    "Training rows:",
    len(train_output)
)

print(
    "Testing rows:",
    len(test_output)
)

print(
    "\nTraining groups:",
    len(train_groups)
)

print(
    "Testing groups:",
    len(test_groups)
)


# Check group overlap

overlap = set(train_groups).intersection(
    set(test_groups)
)

print(
    "\nTrain/test group overlap:",
    len(overlap)
)


print("\nTraining target statistics:")

print(
    train_output[
        target_column
    ].describe()
)


print("\nTesting target statistics:")

print(
    test_output[
        target_column
    ].describe()
)


print("\nTraining file:")

print(TRAIN_PATH)


print("\nTesting file:")

print(TEST_PATH)