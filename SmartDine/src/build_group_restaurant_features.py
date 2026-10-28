import pandas as pd
import numpy as np
import os


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESTAURANT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "restaurant_features.csv"
)

GROUP_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "group_members.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "group_restaurant_features.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("Loading datasets...")

restaurants = pd.read_csv(RESTAURANT_PATH)

groups = pd.read_csv(GROUP_PATH)

print("Restaurants:", restaurants.shape)

print("Group members:", groups.shape)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def parse_cuisines(value):

    if pd.isna(value):
        return set()

    return {
        cuisine.strip().lower()
        for cuisine in str(value).split(",")
        if cuisine.strip()
    }


def cuisine_match(preferred, restaurant_cuisines):

    preferred_set = parse_cuisines(preferred)

    restaurant_set = parse_cuisines(
        restaurant_cuisines
    )

    if not preferred_set:
        return 0.0

    return len(
        preferred_set.intersection(
            restaurant_set
        )
    ) / len(preferred_set)


def budget_match(member_budget, restaurant_cost):

    if pd.isna(restaurant_cost):
        return 0.5

    # Restaurant cost comfortably within budget
    if restaurant_cost <= member_budget:
        return 1.0

    # Slightly above budget
    if restaurant_cost <= member_budget * 1.25:
        return 0.75

    # Moderately above budget
    if restaurant_cost <= member_budget * 1.50:
        return 0.50

    # Significantly above budget
    if restaurant_cost <= member_budget * 2.0:
        return 0.25

    return 0.0


def dietary_match(dietary_preference, restaurant_cuisines):

    preference = str(
        dietary_preference
    ).strip().lower()

    cuisines = parse_cuisines(
        restaurant_cuisines
    )

    if preference == "no preference":
        return 1.0

    # --------------------------------------------------------
    # This is a heuristic because the original dataset does
    # not contain a dedicated dietary-options column.
    # --------------------------------------------------------

    non_veg_keywords = {
        "seafood",
        "biryani",
        "kebab",
        "kabab",
        "meat",
        "steak",
        "fish",
        "chicken",
        "mughlai"
    }

    cuisine_text = " ".join(cuisines)

    contains_nonveg_indicator = any(
        keyword in cuisine_text
        for keyword in non_veg_keywords
    )

    if preference == "vegetarian":

        if contains_nonveg_indicator:
            return 0.5

        return 1.0

    if preference == "vegan":

        if contains_nonveg_indicator:
            return 0.25

        return 0.75

    if preference == "non-vegetarian":

        if contains_nonveg_indicator:
            return 1.0

        return 0.75

    return 0.5


def text_match(preferred_value, restaurant_value):

    if pd.isna(preferred_value):
        return 0.0

    if pd.isna(restaurant_value):
        return 0.0

    preferred = str(
        preferred_value
    ).strip().lower()

    restaurant_values = [
        value.strip().lower()
        for value in str(
            restaurant_value
        ).split(",")
    ]

    if preferred in restaurant_values:
        return 1.0

    # Partial match
    if any(
        preferred in value or value in preferred
        for value in restaurant_values
    ):
        return 0.5

    return 0.0


# ============================================================
# 5. PREPARE RESTAURANT FEATURES
# ============================================================

print("\nPreparing restaurant features...")

restaurants["rating_score"] = (
    restaurants["rating"]
    .fillna(
        restaurants["rating"].median()
    ) / 5.0
)

restaurants["popularity_score"] = (
    restaurants["log_votes"] /
    restaurants["log_votes"].max()
)

restaurants["cost_for_two"] = pd.to_numeric(
    restaurants["cost_for_two"],
    errors="coerce"
)


# ============================================================
# 6. BUILD GROUP × RESTAURANT FEATURES
# ============================================================

print("\nBuilding group × restaurant features...")

rows = []


for group_id, group_members in groups.groupby(
    "group_id"
):

    members = group_members.to_dict(
        "records"
    )

    group_size = len(members)


    # --------------------------------------------------------
    # Group-level agreement
    # --------------------------------------------------------

    cuisine_sets = [
        parse_cuisines(
            member["preferred_cuisines"]
        )
        for member in members
    ]

    pairwise_scores = []

    for i in range(len(cuisine_sets)):

        for j in range(
            i + 1,
            len(cuisine_sets)
        ):

            a = cuisine_sets[i]
            b = cuisine_sets[j]

            union = a.union(b)

            if len(union) == 0:
                similarity = 0.0
            else:
                similarity = (
                    len(a.intersection(b))
                    / len(union)
                )

            pairwise_scores.append(
                similarity
            )

    if pairwise_scores:
        group_agreement = np.mean(
            pairwise_scores
        )
    else:
        group_agreement = 0.0


    # --------------------------------------------------------
    # Evaluate every restaurant
    # --------------------------------------------------------

    for _, restaurant in restaurants.iterrows():

        cuisine_scores = []
        budget_scores = []
        dietary_scores = []
        type_scores = []
        location_scores = []
        booking_scores = []
        online_scores = []


        # ----------------------------------------------------
        # Evaluate each member
        # ----------------------------------------------------

        for member in members:

            # Cuisine
            cuisine_scores.append(
                cuisine_match(
                    member["preferred_cuisines"],
                    restaurant["cuisines"]
                )
            )


            # Budget
            budget_scores.append(
                budget_match(
                    member["budget"],
                    restaurant["cost_for_two"]
                )
            )


            # Dietary
            dietary_scores.append(
                dietary_match(
                    member["dietary_preference"],
                    restaurant["cuisines"]
                )
            )


            # Restaurant type
            type_scores.append(
                text_match(
                    member[
                        "preferred_restaurant_type"
                    ],
                    restaurant["rest_type"]
                )
            )


            # Location
            location_scores.append(
                1.0
                if str(
                    member[
                        "preferred_location"
                    ]
                ).strip().lower()
                ==
                str(
                    restaurant["location"]
                ).strip().lower()
                else 0.0
            )


            # Table booking
            if member[
                "requires_table_booking"
            ] == 1:

                booking_scores.append(
                    float(
                        restaurant["book_table"]
                    )
                )

            else:

                booking_scores.append(1.0)


            # Online ordering
            if member[
                "prefers_online_order"
            ] == 1:

                online_scores.append(
                    float(
                        restaurant["online_order"]
                    )
                )

            else:

                online_scores.append(1.0)


        # ====================================================
        # GROUP-LEVEL AGGREGATION
        # ====================================================

        cuisine_match_score = np.mean(
            cuisine_scores
        )

        budget_match_score = np.mean(
            budget_scores
        )

        dietary_compatibility = np.mean(
            dietary_scores
        )

        restaurant_type_match = np.mean(
            type_scores
        )

        location_match = np.mean(
            location_scores
        )

        booking_match = np.mean(
            booking_scores
        )

        online_order_match = np.mean(
            online_scores
        )


        # ====================================================
        # SUITABILITY SCORE
        # ====================================================

        # Weighted combination of group compatibility
        # and restaurant quality/context.

        suitability = (
            0.20 * cuisine_match_score
            + 0.20 * budget_match_score
            + 0.15 * dietary_compatibility
            + 0.10 * restaurant_type_match
            + 0.10 * location_match
            + 0.05 * booking_match
            + 0.05 * online_order_match
            + 0.10 * restaurant[
                "rating_score"
            ]
            + 0.05 * restaurant[
                "popularity_score"
            ]
        )


        # ----------------------------------------------------
        # Add controlled noise
        # ----------------------------------------------------

        noise = np.random.normal(
            loc=0.0,
            scale=0.03
        )

        suitability = suitability + noise

        suitability = np.clip(
            suitability,
            0.0,
            1.0
        )


        # ====================================================
        # STORE RESULT
        # ====================================================

        rows.append({

            "group_id":
                group_id,

            "restaurant_id":
                restaurant[
                    "restaurant_id"
                ],

            "group_size":
                group_size,

            "cuisine_match":
                cuisine_match_score,

            "budget_match":
                budget_match_score,

            "dietary_compatibility":
                dietary_compatibility,

            "restaurant_type_match":
                restaurant_type_match,

            "location_match":
                location_match,

            "booking_match":
                booking_match,

            "online_order_match":
                online_order_match,

            "group_agreement":
                group_agreement,

            "rating_score":
                restaurant[
                    "rating_score"
                ],

            "popularity_score":
                restaurant[
                    "popularity_score"
                ],

            "rating":
                restaurant[
                    "rating"
                ],

            "cost_for_two":
                restaurant[
                    "cost_for_two"
                ],

            "votes":
                restaurant[
                    "votes"
                ],

            "online_order":
                restaurant[
                    "online_order"
                ],

            "book_table":
                restaurant[
                    "book_table"
                ],

            "suitability_score":
                suitability
        })


# ============================================================
# 7. CREATE DATAFRAME
# ============================================================

group_restaurant = pd.DataFrame(
    rows
)


# ============================================================
# 8. SAVE DATASET
# ============================================================

print("\nSaving group × restaurant dataset...")

group_restaurant.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 9. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("GROUP × RESTAURANT FEATURE ENGINEERING COMPLETED")
print("=" * 70)

print(
    "Number of groups:",
    group_restaurant["group_id"].nunique()
)

print(
    "Number of restaurants:",
    group_restaurant["restaurant_id"].nunique()
)

print(
    "Total combinations:",
    len(group_restaurant)
)

print(
    "\nExpected combinations:",
    groups["group_id"].nunique()
    * restaurants["restaurant_id"].nunique()
)

print("\nFeature columns:")

print(
    group_restaurant.columns.tolist()
)

print("\nMissing values:")

print(
    group_restaurant
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
)

print("\nSuitability score statistics:")

print(
    group_restaurant[
        "suitability_score"
    ].describe()
)

print("\nFirst 10 rows:")

print(
    group_restaurant
    .head(10)
    .to_string(
        index=False
    )
)

print("\nOutput file:")

print(OUTPUT_PATH)