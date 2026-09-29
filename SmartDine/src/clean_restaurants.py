import pandas as pd
import numpy as np
import os


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RAW_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "zomato.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "restaurants_cleaned.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading raw dataset...")

df = pd.read_csv(RAW_PATH)

print(f"Raw dataset shape: {df.shape}")


# ============================================================
# 3. BASIC TEXT CLEANING
# ============================================================

text_columns = [
    "name",
    "address",
    "location",
    "rest_type",
    "dish_liked",
    "cuisines",
    "reviews_list",
    "menu_item",
    "listed_in(type)",
    "listed_in(city)"
]

for column in text_columns:
    if column in df.columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )


# ============================================================
# 4. CLEAN RATING
# ============================================================

# Extract numeric rating from values such as:
# 4.1/5
# 4.1 /5
# NEW
# -
# NaN

df["rating"] = (
    df["rate"]
    .astype("string")
    .str.extract(r"(\d+(?:\.\d+)?)")[0]
)

df["rating"] = pd.to_numeric(
    df["rating"],
    errors="coerce"
)


# ============================================================
# 5. CLEAN COST
# ============================================================

df["cost_for_two"] = (
    df["approx_cost(for two people)"]
    .astype("string")
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["cost_for_two"] = pd.to_numeric(
    df["cost_for_two"],
    errors="coerce"
)


# ============================================================
# 6. CLEAN YES/NO FEATURES
# ============================================================

df["online_order"] = (
    df["online_order"]
    .astype("string")
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "no": 0
    })
)

df["book_table"] = (
    df["book_table"]
    .astype("string")
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "no": 0
    })
)


# ============================================================
# 7. CREATE RESTAURANT ENTITY ID
# ============================================================

# name + address currently identifies the physical
# restaurant entity for our consolidation step.

df["restaurant_id"] = (
    df["name"].astype("string").str.strip()
    + " | "
    + df["address"].astype("string").str.strip()
)


# ============================================================
# 8. HELPER FUNCTIONS
# ============================================================

def combine_unique_values(series):
    """
    Combine unique non-null values into one comma-separated string.
    """

    values = []

    for value in series.dropna():

        value = str(value).strip()

        if value and value.lower() not in ["nan", "none", "-"]:

            if value not in values:
                values.append(value)

    return ", ".join(values)


def first_valid(series):
    """
    Return the first valid non-null value.
    """

    valid = series.dropna()

    if len(valid) == 0:
        return np.nan

    return valid.iloc[0]


def median_valid(series):
    """
    Return median of valid numeric values.
    """

    valid = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(valid) == 0:
        return np.nan

    return valid.median()


# ============================================================
# 9. CONSOLIDATE RESTAURANT ENTITIES
# ============================================================

print("Consolidating restaurant entities...")

restaurant_df = (
    df.groupby(
        ["restaurant_id", "name", "address"],
        dropna=False
    )
    .agg(

        # Geographic information
        location=("location", first_valid),

        # Restaurant characteristics
        cuisines=("cuisines", combine_unique_values),
        rest_type=("rest_type", combine_unique_values),

        # Listing/context information
        listed_in_city=("listed_in(city)", combine_unique_values),
        listed_in_type=("listed_in(type)", combine_unique_values),

        # Numeric restaurant attributes
        rating=("rating", median_valid),
        votes=("votes", "max"),
        cost_for_two=("cost_for_two", median_valid),

        # Service features
        online_order=("online_order", "max"),
        book_table=("book_table", "max"),

        # Optional text fields
        dish_liked=("dish_liked", combine_unique_values),
        menu_item=("menu_item", combine_unique_values),

        # Number of raw records represented
        source_record_count=("restaurant_id", "size")
    )
    .reset_index()
)


# ============================================================
# 10. CLEAN SPARSE TEXT FIELDS
# ============================================================

# Convert empty-list representations to missing values.
# These fields are retained for possible future NLP analysis,
# but will not be used as core Phase-1 recommendation features.

restaurant_df["menu_item"] = (
    restaurant_df["menu_item"]
    .replace("[]", np.nan)
)

restaurant_df["dish_liked"] = (
    restaurant_df["dish_liked"]
    .replace("", np.nan)
)


# ============================================================
# 11. CLEAN MOJIBAKE / ENCODING ISSUES
# ============================================================

def fix_mojibake(text):
    """
    Attempt to repair common UTF-8/Latin-1 mojibake.
    If repair fails, retain the original value.
    """

    if pd.isna(text):
        return text

    text = str(text)

    try:
        if "Ã" in text or "Â" in text:
            repaired = text.encode("latin1").decode("utf-8")

            # Only use repaired value if it improves
            # the obvious mojibake pattern.
            if repaired.count("Ã") + repaired.count("Â") < \
               text.count("Ã") + text.count("Â"):
                return repaired

    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    return text


for column in ["name", "address", "location", "cuisines", "rest_type"]:

    restaurant_df[column] = (
        restaurant_df[column]
        .apply(fix_mojibake)
    )


# ============================================================
# 12. FINAL DATA TYPES
# ============================================================

restaurant_df["rating"] = pd.to_numeric(
    restaurant_df["rating"],
    errors="coerce"
)

restaurant_df["votes"] = pd.to_numeric(
    restaurant_df["votes"],
    errors="coerce"
)

restaurant_df["cost_for_two"] = pd.to_numeric(
    restaurant_df["cost_for_two"],
    errors="coerce"
)

restaurant_df["online_order"] = pd.to_numeric(
    restaurant_df["online_order"],
    errors="coerce"
)

restaurant_df["book_table"] = pd.to_numeric(
    restaurant_df["book_table"],
    errors="coerce"
)


# ============================================================
# 13. REMOVE DUPLICATE RESTAURANT ENTITIES
# ============================================================

restaurant_df = (
    restaurant_df
    .drop_duplicates(subset=["restaurant_id"])
    .reset_index(drop=True)
)


# ============================================================
# 14. SORT DATA
# ============================================================

restaurant_df = restaurant_df.sort_values(
    by="name",
    na_position="last"
).reset_index(drop=True)


# ============================================================
# 15. SAVE CLEANED DATASET
# ============================================================

restaurant_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 16. FINAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("CLEANING COMPLETED")
print("=" * 70)

print(f"Raw records: {len(df):,}")
print(f"Restaurant entities: {len(restaurant_df):,}")
print(f"Output shape: {restaurant_df.shape}")

print()
print("Duplicate restaurant IDs:",
      restaurant_df["restaurant_id"].duplicated().sum())

print()
print("Missing values:")
print(
    restaurant_df.isna()
    .sum()
    .sort_values(ascending=False)
)

print()
print("Output file:")
print(OUTPUT_PATH)

print()
print("Rating statistics:")
print(
    restaurant_df["rating"].describe()
)

print()
print("Cost statistics:")
print(
    restaurant_df["cost_for_two"].describe()
)