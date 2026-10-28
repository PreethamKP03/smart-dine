"""
SmartDine — Data Preprocessing Module
src/preprocessing.py

Responsibilities:
1. Load raw restaurant dataset (zomato.csv)
2. Clean text, rating, and cost fields
3. Consolidate duplicate records into physical restaurant entities
4. Save cleaned dataset to data/processed/restaurants_cleaned.csv
"""

import os
import re
import numpy as np
import pandas as pd


def get_project_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "data", "raw", "zomato.csv")
    output_dir = os.path.join(base_dir, "data", "processed")
    output_path = os.path.join(output_dir, "restaurants_cleaned.csv")
    os.makedirs(output_dir, exist_ok=True)
    return raw_path, output_dir, output_path


def combine_unique_values(series):
    """Combine unique non-null string values separated by comma."""
    values = []
    for value in series.dropna():
        for item in str(value).split(","):
            cleaned = item.strip()
            if cleaned and cleaned not in values:
                values.append(cleaned)
    return ", ".join(values) if values else np.nan


def clean_dataset(raw_path=None, output_path=None):
    default_raw, _, default_out = get_project_paths()
    raw_path = raw_path or default_raw
    output_path = output_path or default_out

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at: {raw_path}")

    print(f"Loading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"Raw shape: {df.shape}")

    # 1. Clean Text Columns
    text_columns = [
        "name", "address", "location", "rest_type", "dish_liked",
        "cuisines", "reviews_list", "menu_item", "listed_in(type)", "listed_in(city)"
    ]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # 2. Clean Rating
    df["rating"] = df["rate"].astype("string").str.extract(r"(\d+(?:\.\d+)?)")[0]
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # 3. Clean Cost for Two
    df["cost_for_two"] = (
        df["approx_cost(for two people)"]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["cost_for_two"] = pd.to_numeric(df["cost_for_two"], errors="coerce")

    # 4. Clean Boolean Flags
    df["online_order"] = df["online_order"].astype("string").str.strip().str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)
    df["book_table"] = df["book_table"].astype("string").str.strip().str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)

    # 5. Clean Votes
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)

    # 6. Create Unique Entity Identifier (Name + Address)
    df["restaurant_id"] = (
        df["name"].astype("string").str.strip()
        + " | "
        + df["address"].astype("string").str.strip()
    )

    # 7. Consolidate Multiple Duplicate Listing Records for Same Physical Restaurant
    print("Consolidating duplicate records into restaurant entities...")
    restaurant_df = df.groupby("restaurant_id", as_index=False).agg({
        "name": "first",
        "address": "first",
        "location": "first",
        "rest_type": combine_unique_values,
        "dish_liked": combine_unique_values,
        "cuisines": combine_unique_values,
        "listed_in(type)": combine_unique_values,
        "listed_in(city)": combine_unique_values,
        "menu_item": combine_unique_values,
        "rating": "median",
        "cost_for_two": "median",
        "votes": "max",
        "online_order": "max",
        "book_table": "max",
        "rate": "count"
    }).rename(columns={
        "listed_in(type)": "listed_in_type",
        "listed_in(city)": "listed_in_city",
        "rate": "source_record_count"
    })

    # Sort and Save
    restaurant_df = restaurant_df.sort_values(by="name", na_position="last").reset_index(drop=True)
    restaurant_df.to_csv(output_path, index=False)

    print("=" * 60)
    print("DATA PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Raw records: {len(df):,}")
    print(f"Consolidated entities: {len(restaurant_df):,}")
    print(f"Saved cleaned data to: {output_path}")

    return restaurant_df


if __name__ == "__main__":
    clean_dataset()
