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
    "restaurant_features.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "group_members.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_SEED = 42
NUMBER_OF_GROUPS = 100

np.random.seed(RANDOM_SEED)


# ============================================================
# 3. LOAD RESTAURANT DATA
# ============================================================

print("Loading restaurant feature dataset...")

restaurants = pd.read_csv(INPUT_PATH)

print("Restaurant dataset shape:", restaurants.shape)


# ============================================================
# 4. EXTRACT REAL VALUES FROM DATASET
# ============================================================

# ------------------------------------------------------------
# Cuisines
# ------------------------------------------------------------

cuisine_counts = {}

for value in restaurants["cuisines"].dropna():

    cuisines = [
        cuisine.strip()
        for cuisine in str(value).split(",")
        if cuisine.strip()
    ]

    for cuisine in cuisines:
        cuisine_counts[cuisine] = (
            cuisine_counts.get(cuisine, 0) + 1
        )


# Use reasonably common cuisines
cuisine_series = (
    pd.Series(cuisine_counts)
    .sort_values(ascending=False)
)

available_cuisines = cuisine_series.head(30).index.tolist()


# ------------------------------------------------------------
# Restaurant types
# ------------------------------------------------------------

rest_type_counts = (
    restaurants["rest_type"]
    .dropna()
    .value_counts()
)

available_restaurant_types = (
    rest_type_counts.head(15)
    .index
    .tolist()
)


# ------------------------------------------------------------
# Locations
# ------------------------------------------------------------

location_counts = (
    restaurants["location"]
    .dropna()
    .value_counts()
)

available_locations = (
    location_counts.head(20)
    .index
    .tolist()
)


# ============================================================
# 5. VALIDATE AVAILABLE VALUES
# ============================================================

print("\nAvailable preference categories:")

print(
    "Cuisines:",
    len(available_cuisines)
)

print(
    "Restaurant types:",
    len(available_restaurant_types)
)

print(
    "Locations:",
    len(available_locations)
)


# ============================================================
# 6. PREFERENCE OPTIONS
# ============================================================

dietary_options = [
    "No Preference",
    "Vegetarian",
    "Non-Vegetarian",
    "Vegan"
]


# ============================================================
# 7. GENERATE GROUP MEMBERS
# ============================================================

print("\nGenerating group preference data...")

rows = []

for group_number in range(
    1,
    NUMBER_OF_GROUPS + 1
):

    # --------------------------------------------------------
    # Random group size: 3–5 members
    # --------------------------------------------------------

    group_size = np.random.randint(3, 6)

    group_id = f"G{group_number:03d}"


    # --------------------------------------------------------
    # Generate members
    # --------------------------------------------------------

    for member_number in range(
        1,
        group_size + 1
    ):

        member_id = (
            f"{group_id}_M{member_number}"
        )


        # ----------------------------------------------------
        # Cuisine preference
        # ----------------------------------------------------

        number_of_cuisines = np.random.choice(
            [1, 2],
            p=[0.70, 0.30]
        )

        preferred_cuisines = np.random.choice(
            available_cuisines,
            size=number_of_cuisines,
            replace=False
        )

        preferred_cuisines = ", ".join(
            preferred_cuisines
        )


        # ----------------------------------------------------
        # Budget
        # ----------------------------------------------------

        budget_options = [
            300,
            400,
            500,
            600,
            750,
            1000,
            1500,
            2000
        ]

        budget = np.random.choice(
            budget_options
        )


        # ----------------------------------------------------
        # Dietary preference
        # ----------------------------------------------------

        dietary_preference = np.random.choice(
            dietary_options,
            p=[0.35, 0.40, 0.15, 0.10]
        )


        # ----------------------------------------------------
        # Restaurant type preference
        # ----------------------------------------------------

        preferred_restaurant_type = np.random.choice(
            available_restaurant_types
        )


        # ----------------------------------------------------
        # Location preference
        # ----------------------------------------------------

        preferred_location = np.random.choice(
            available_locations
        )


        # ----------------------------------------------------
        # Service preferences
        # ----------------------------------------------------

        requires_table_booking = np.random.choice(
            [0, 1],
            p=[0.70, 0.30]
        )

        prefers_online_order = np.random.choice(
            [0, 1],
            p=[0.45, 0.55]
        )


        # ----------------------------------------------------
        # Preference importance weights
        # ----------------------------------------------------

        weights = np.random.dirichlet(
            np.ones(5)
        )

        cuisine_importance = round(
            weights[0],
            3
        )

        budget_importance = round(
            weights[1],
            3
        )

        dietary_importance = round(
            weights[2],
            3
        )

        restaurant_type_importance = round(
            weights[3],
            3
        )

        location_importance = round(
            weights[4],
            3
        )


        # ----------------------------------------------------
        # Store member
        # ----------------------------------------------------

        rows.append({
            "group_id": group_id,
            "member_id": member_id,
            "preferred_cuisines": preferred_cuisines,
            "budget": budget,
            "dietary_preference": dietary_preference,
            "preferred_restaurant_type":
                preferred_restaurant_type,
            "preferred_location":
                preferred_location,
            "requires_table_booking":
                requires_table_booking,
            "prefers_online_order":
                prefers_online_order,
            "cuisine_importance":
                cuisine_importance,
            "budget_importance":
                budget_importance,
            "dietary_importance":
                dietary_importance,
            "restaurant_type_importance":
                restaurant_type_importance,
            "location_importance":
                location_importance
        })


# ============================================================
# 8. CREATE DATAFRAME
# ============================================================

group_members = pd.DataFrame(rows)


# ============================================================
# 9. SAVE DATASET
# ============================================================

group_members.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 10. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("GROUP DATA GENERATION COMPLETED")
print("=" * 70)

print(
    "Number of groups:",
    group_members["group_id"].nunique()
)

print(
    "Number of members:",
    group_members["member_id"].nunique()
)

print(
    "Total rows:",
    len(group_members)
)

print("\nGroup size distribution:")

group_sizes = (
    group_members
    .groupby("group_id")
    .size()
    .value_counts()
    .sort_index()
)

print(group_sizes)


print("\nDataset columns:")

print(group_members.columns.tolist())


print("\nFirst 10 records:")

print(
    group_members
    .head(10)
    .to_string(index=False)
)


print("\nMissing values:")

print(
    group_members
    .isna()
    .sum()
)


print("\nOutput file:")

print(OUTPUT_PATH)