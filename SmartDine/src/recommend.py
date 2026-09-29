# src/recommend.py

import os
import re
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

RESTAURANT_PATH = "data/processed/restaurant_features.csv"
GROUP_PATH = "data/processed/group_members.csv"
MODEL_PATH = "models/smartdine_best_model.joblib"
FEATURE_PATH = "models/feature_columns.txt"

OUTPUT_DIR = "outputs"

TOP_N = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading SmartDine data...")

restaurants = pd.read_csv(RESTAURANT_PATH)
groups = pd.read_csv(GROUP_PATH)

model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH, "r") as f:
    feature_columns = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"Restaurants loaded: {len(restaurants):,}")
print(f"Groups loaded: {groups['group_id'].nunique():,}")
print(f"Model loaded: {model.__class__.__name__}")


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = value.replace("&", "and")

    value = re.sub(r"\s+", " ", value)

    return value


def split_values(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        normalize_text(item)
        for item in value.split(",")
        if normalize_text(item)
    ]


# ============================================================
# MEMBER-LEVEL COMPATIBILITY
# ============================================================

def cuisine_match(
    preferred_cuisines,
    restaurant_cuisines
):

    preferred = set(
        split_values(preferred_cuisines)
    )

    available = set(
        split_values(restaurant_cuisines)
    )

    if not preferred:
        return 0.5

    if not available:
        return 0.0

    return min(
        1.0,
        len(
            preferred.intersection(available)
        ) / len(preferred)
    )


def budget_match(
    member_budget,
    restaurant_cost
):

    if pd.isna(member_budget):
        return 0.5

    if pd.isna(restaurant_cost):
        return 0.5

    member_budget = float(member_budget)

    if member_budget <= 0:
        return 0.5

    ratio = (
        restaurant_cost /
        member_budget
    )

    if ratio <= 1.0:
        return 1.0

    if ratio <= 1.25:
        return 0.75

    if ratio <= 1.5:
        return 0.50

    if ratio <= 2.0:
        return 0.25

    return 0.0


def dietary_match(
    preference,
    cuisines
):

    preference = normalize_text(
        preference
    )

    if (
        not preference
        or preference == "no preference"
    ):
        return 1.0

    cuisine_text = " ".join(
        split_values(cuisines)
    )

    non_veg_terms = [
        "chicken",
        "mutton",
        "fish",
        "meat",
        "seafood",
        "pork",
        "beef"
    ]

    animal_product_terms = [
        "chicken",
        "mutton",
        "fish",
        "meat",
        "seafood",
        "pork",
        "beef",
        "egg",
        "cheese",
        "butter",
        "milk",
        "cream"
    ]

    if preference == "vegetarian":

        if any(
            term in cuisine_text
            for term in non_veg_terms
        ):
            return 0.4

        return 1.0

    if preference == "vegan":

        if any(
            term in cuisine_text
            for term in animal_product_terms
        ):
            return 0.3

        return 1.0

    if preference == "non-vegetarian":
        return 1.0

    return 0.5


def type_match(
    preferred_type,
    restaurant_type
):

    preferred_type = normalize_text(
        preferred_type
    )

    restaurant_types = set(
        split_values(restaurant_type)
    )

    if not preferred_type:
        return 0.5

    if not restaurant_types:
        return 0.0

    return (
        1.0
        if preferred_type in restaurant_types
        else 0.0
    )


def location_match(
    preferred_location,
    restaurant_location
):

    preferred_location = normalize_text(
        preferred_location
    )

    restaurant_location = normalize_text(
        restaurant_location
    )

    if not preferred_location:
        return 0.5

    if not restaurant_location:
        return 0.0

    return (
        1.0
        if preferred_location == restaurant_location
        else 0.0
    )


# ============================================================
# GROUP AGREEMENT
# ============================================================

def calculate_group_agreement(group):

    agreement_values = []

    dimensions = [
        "preferred_cuisines",
        "preferred_restaurant_type",
        "preferred_location",
        "budget",
        "dietary_preference"
    ]

    for column in dimensions:

        values = group[column].astype(
            str
        ).map(normalize_text)

        if len(values) == 0:
            continue

        concentration = (
            values.value_counts().max()
            / len(values)
        )

        agreement_values.append(
            concentration
        )

    if not agreement_values:
        return 0.5

    return float(
        np.mean(agreement_values)
    )


# ============================================================
# WEIGHTED GROUP PROFILE
# ============================================================

def aggregate_group_preferences(group):

    # --------------------------------------------------------
    # Cuisine
    # --------------------------------------------------------

    cuisine_scores = {}

    for _, member in group.iterrows():

        cuisines = split_values(
            member["preferred_cuisines"]
        )

        weight = float(
            member.get(
                "cuisine_importance",
                1.0
            )
        )

        for cuisine in cuisines:

            cuisine_scores[cuisine] = (
                cuisine_scores.get(
                    cuisine,
                    0.0
                )
                + weight
            )

    total = sum(
        cuisine_scores.values()
    )

    if total > 0:

        cuisine_scores = {
            cuisine: score / total
            for cuisine, score
            in cuisine_scores.items()
        }

    ranked_cuisines = sorted(
        cuisine_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    group_cuisines = [
        cuisine
        for cuisine, _ in ranked_cuisines[:5]
    ]


    # --------------------------------------------------------
    # Budget
    # --------------------------------------------------------

    budgets = pd.to_numeric(
        group["budget"],
        errors="coerce"
    )

    budget_weights = pd.to_numeric(
        group["budget_importance"],
        errors="coerce"
    )

    valid = (
        budgets.notna()
        & budget_weights.notna()
    )

    if valid.any():

        group_budget = np.average(
            budgets[valid],
            weights=budget_weights[valid]
        )

    else:

        group_budget = 500.0


    # --------------------------------------------------------
    # Dietary
    # --------------------------------------------------------

    dietary_values = [
        normalize_text(value)
        for value in group[
            "dietary_preference"
        ].dropna()
    ]

    specific = [
        value
        for value in dietary_values
        if value != "no preference"
    ]

    group_dietary = list(
        dict.fromkeys(specific)
    )

    if not group_dietary:
        group_dietary = [
            "no preference"
        ]


    # --------------------------------------------------------
    # Restaurant Type
    # --------------------------------------------------------

    type_scores = {}

    for _, member in group.iterrows():

        value = normalize_text(
            member[
                "preferred_restaurant_type"
            ]
        )

        weight = float(
            member.get(
                "restaurant_type_importance",
                1.0
            )
        )

        if value:

            type_scores[value] = (
                type_scores.get(
                    value,
                    0.0
                )
                + weight
            )

    ranked_types = sorted(
        type_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    group_types = [
        item[0]
        for item in ranked_types[:5]
    ]


    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    location_scores = {}

    for _, member in group.iterrows():

        value = normalize_text(
            member[
                "preferred_location"
            ]
        )

        weight = float(
            member.get(
                "location_importance",
                1.0
            )
        )

        if value:

            location_scores[value] = (
                location_scores.get(
                    value,
                    0.0
                )
                + weight
            )

    ranked_locations = sorted(
        location_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    group_locations = [
        item[0]
        for item in ranked_locations[:5]
    ]


    # --------------------------------------------------------
    # Booking
    # --------------------------------------------------------

    booking_values = pd.to_numeric(
        group[
            "requires_table_booking"
        ],
        errors="coerce"
    ).fillna(0)

    booking_required = bool(
        booking_values.mean() >= 0.5
    )


    # --------------------------------------------------------
    # Online ordering
    # --------------------------------------------------------

    online_values = pd.to_numeric(
        group[
            "prefers_online_order"
        ],
        errors="coerce"
    ).fillna(0)

    online_preferred = bool(
        online_values.mean() >= 0.5
    )


    # --------------------------------------------------------
    # Agreement
    # --------------------------------------------------------

    agreement = calculate_group_agreement(
        group
    )


    return {
        "cuisines": group_cuisines,
        "budget": float(group_budget),
        "dietary": group_dietary,
        "restaurant_types": group_types,
        "locations": group_locations,
        "booking_required":
            booking_required,
        "online_preferred":
            online_preferred,
        "group_agreement":
            agreement
    }


# ============================================================
# BUILD GROUP-RESTAURANT FEATURES
# ============================================================

def build_features(
    group,
    profile,
    restaurant
):

    cuisine_scores = []
    cuisine_weights = []

    budget_scores = []
    budget_weights = []

    dietary_scores = []
    dietary_weights = []

    type_scores = []
    type_weights = []

    location_scores = []
    location_weights = []


    # --------------------------------------------------------
    # Calculate member-level scores
    # --------------------------------------------------------

    for _, member in group.iterrows():

        # Cuisine
        cuisine_scores.append(
            cuisine_match(
                member[
                    "preferred_cuisines"
                ],
                restaurant[
                    "cuisines"
                ]
            )
        )

        cuisine_weights.append(
            float(
                member.get(
                    "cuisine_importance",
                    1.0
                )
            )
        )


        # Budget
        budget_scores.append(
            budget_match(
                member["budget"],
                restaurant[
                    "cost_for_two"
                ]
            )
        )

        budget_weights.append(
            float(
                member.get(
                    "budget_importance",
                    1.0
                )
            )
        )


        # Dietary
        dietary_scores.append(
            dietary_match(
                member[
                    "dietary_preference"
                ],
                restaurant[
                    "cuisines"
                ]
            )
        )

        dietary_weights.append(
            float(
                member.get(
                    "dietary_importance",
                    1.0
                )
            )
        )


        # Restaurant type
        type_scores.append(
            type_match(
                member[
                    "preferred_restaurant_type"
                ],
                restaurant[
                    "rest_type"
                ]
            )
        )

        type_weights.append(
            float(
                member.get(
                    "restaurant_type_importance",
                    1.0
                )
            )
        )


        # Location
        location_scores.append(
            location_match(
                member[
                    "preferred_location"
                ],
                restaurant[
                    "location"
                ]
            )
        )

        location_weights.append(
            float(
                member.get(
                    "location_importance",
                    1.0
                )
            )
        )


    # --------------------------------------------------------
    # Weighted average helper
    # --------------------------------------------------------

    def weighted_average(
        values,
        weights
    ):

        if not values:
            return 0.5

        total_weight = sum(
            weights
        )

        if total_weight <= 0:
            return float(
                np.mean(values)
            )

        return float(
            np.average(
                values,
                weights=weights
            )
        )


    cuisine_score = weighted_average(
        cuisine_scores,
        cuisine_weights
    )

    budget_score = weighted_average(
        budget_scores,
        budget_weights
    )

    dietary_score = weighted_average(
        dietary_scores,
        dietary_weights
    )

    type_score = weighted_average(
        type_scores,
        type_weights
    )

    location_score = weighted_average(
        location_scores,
        location_weights
    )


    # --------------------------------------------------------
    # Booking
    # --------------------------------------------------------

    if profile["booking_required"]:

        booking_score = (
            1.0
            if restaurant["book_table"] == 1
            else 0.0
        )

    else:

        booking_score = 0.5


    # --------------------------------------------------------
    # Online ordering
    # --------------------------------------------------------

    if profile["online_preferred"]:

        online_score = (
            1.0
            if restaurant["online_order"] == 1
            else 0.0
        )

    else:

        online_score = 0.5


    # --------------------------------------------------------
    # Rating
    # --------------------------------------------------------

    if pd.isna(
        restaurant["rating"]
    ):

        rating_score = 0.5

    else:

        rating_score = np.clip(
            (
                restaurant["rating"] - 1
            ) / 4,
            0,
            1
        )


    # --------------------------------------------------------
    # Popularity
    # --------------------------------------------------------

    votes = restaurant["votes"]

    if pd.isna(votes):

        popularity_score = 0.0

    else:

        popularity_score = (
            np.log1p(
                max(0, votes)
            )
            /
            np.log1p(
                max(
                    1,
                    restaurants[
                        "votes"
                    ].max()
                )
            )
        )


    # --------------------------------------------------------
    # Feature dictionary
    # --------------------------------------------------------

    return {

        "group_size":
            len(group),

        "cuisine_match":
            cuisine_score,

        "budget_match":
            budget_score,

        "dietary_compatibility":
            dietary_score,

        "restaurant_type_match":
            type_score,

        "location_match":
            location_score,

        "booking_match":
            booking_score,

        "online_order_match":
            online_score,

        "group_agreement":
            profile[
                "group_agreement"
            ],

        "rating_score":
            rating_score,

        "popularity_score":
            popularity_score,

        "cost_for_two":
            (
                restaurant[
                    "cost_for_two"
                ]
                if not pd.isna(
                    restaurant[
                        "cost_for_two"
                    ]
                )
                else restaurants[
                    "cost_for_two"
                ].median()
            ),

        "votes":
            (
                restaurant["votes"]
                if not pd.isna(
                    restaurant["votes"]
                )
                else 0
            ),

        "online_order":
            restaurant[
                "online_order"
            ],

        "book_table":
            restaurant[
                "book_table"
            ]
    }


# ============================================================
# HARD CONSTRAINTS
# ============================================================

def passes_constraints(
    group,
    profile,
    restaurant
):

    # --------------------------------------------------------
    # Budget constraint
    # --------------------------------------------------------

    budget_importance = pd.to_numeric(
        group[
            "budget_importance"
        ],
        errors="coerce"
    )

    mean_budget_importance = (
        budget_importance.mean()
    )

    if (
        mean_budget_importance >= 0.60
        and not pd.isna(
            restaurant[
                "cost_for_two"
            ]
        )
    ):

        if (
            restaurant[
                "cost_for_two"
            ]
            > profile["budget"] * 1.5
        ):

            return False


    # --------------------------------------------------------
    # Booking constraint
    # --------------------------------------------------------

    if (
        profile["booking_required"]
        and restaurant[
            "book_table"
        ] != 1
    ):

        return False


    return True


# ============================================================
# EXPLANATION
# ============================================================

def generate_explanation(
    row
):

    reasons = []
    cautions = []


    # Cuisine
    if row[
        "cuisine_match"
    ] >= 0.75:

        reasons.append(
            "strong cuisine compatibility"
        )

    elif row[
        "cuisine_match"
    ] >= 0.40:

        reasons.append(
            "partial cuisine compatibility"
        )

    else:

        cautions.append(
            "limited cuisine compatibility"
        )


    # Budget
    if row[
        "budget_match"
    ] >= 0.75:

        reasons.append(
            "within the group's preferred budget"
        )

    elif row[
        "budget_match"
    ] >= 0.40:

        reasons.append(
            "reasonably close to the group's budget"
        )

    else:

        cautions.append(
            "above the preferred budget"
        )


    # Dietary
    if row[
        "dietary_compatibility"
    ] >= 0.75:

        if row[
            "dietary_conflict"
        ]:

            cautions.append(
                "dietary options should be verified"
            )

        else:

            reasons.append(
                "good dietary compatibility"
            )

    else:

        cautions.append(
            "dietary compatibility should be verified"
        )


    # Type
    if row[
        "restaurant_type_match"
    ] >= 0.75:

        reasons.append(
            "matches preferred restaurant type"
        )

    elif row[
        "restaurant_type_match"
    ] >= 0.40:

        reasons.append(
            "partially matches preferred restaurant type"
        )


    # Location
    if row[
        "location_match"
    ] >= 0.75:

        reasons.append(
            "preferred location for most members"
        )

    elif row[
        "location_match"
    ] >= 0.40:

        reasons.append(
            "location matches some group members"
        )


    # Booking
    if row[
        "booking_required"
    ]:

        reasons.append(
            "table booking requirement satisfied"
        )


    # Online
    if row[
        "online_preferred"
    ]:

        reasons.append(
            "online-order preference satisfied"
        )


    # Rating
    if (
        not pd.isna(
            row["rating"]
        )
        and row["rating"] >= 4.0
    ):

        reasons.append(
            f"good rating ({row['rating']:.1f}/5)"
        )


    if reasons:

        explanation = (
            "Recommended because of "
            + ", ".join(reasons)
            + "."
        )

    else:

        explanation = (
            "Recommended based on overall "
            "group-restaurant compatibility."
        )


    if cautions:

        explanation += (
            " Caution: "
            + ", ".join(cautions)
            + "."
        )


    return explanation


# ============================================================
# DIVERSITY-AWARE RANKING
# ============================================================

def apply_diversity(
    results,
    top_n
):

    selected = []

    used_names = set()

    # First pass:
    # one branch per restaurant name

    for _, row in results.iterrows():

        name = normalize_text(
            row["name"]
        )

        if name in used_names:
            continue

        selected.append(row)
        used_names.add(name)

        if len(selected) >= top_n:
            break


    # Second pass:
    # fill remaining slots if necessary

    if len(selected) < top_n:

        for _, row in results.iterrows():

            if any(
                row["restaurant_id"]
                == selected_row[
                    "restaurant_id"
                ]
                for selected_row in selected
            ):
                continue

            selected.append(row)

            if len(selected) >= top_n:
                break


    return pd.DataFrame(
        selected
    )


# ============================================================
# RECOMMEND FOR GROUP
# ============================================================

def recommend_for_group(
    group_id,
    top_n=TOP_N
):

    group = groups[
        groups["group_id"] == group_id
    ].copy()

    if group.empty:

        raise ValueError(
            f"Group '{group_id}' not found."
        )


    print("\n" + "=" * 70)

    print(
        f"SMARTDINE RECOMMENDATIONS — {group_id}"
    )

    print("=" * 70)


    print(
        f"Group size: {len(group)} members"
    )


    # --------------------------------------------------------
    # Profile
    # --------------------------------------------------------

    profile = aggregate_group_preferences(
        group
    )


    print("\nGroup Preferences:")

    print(
        "Preferred cuisines:",
        ", ".join(
            profile["cuisines"]
        )
    )

    print(
        f"Weighted budget: "
        f"₹{profile['budget']:.0f}"
    )

    print(
        "Dietary preferences:",
        ", ".join(
            profile["dietary"]
        )
    )

    print(
        "Restaurant types:",
        ", ".join(
            profile["restaurant_types"]
        )
    )

    print(
        "Preferred locations:",
        ", ".join(
            profile["locations"]
        )
    )

    print(
        f"Group agreement: "
        f"{profile['group_agreement']:.3f}"
    )


    # --------------------------------------------------------
    # Candidate filtering
    # --------------------------------------------------------

    candidates = restaurants[
        restaurants.apply(
            lambda restaurant:
                passes_constraints(
                    group,
                    profile,
                    restaurant
                ),
            axis=1
        )
    ].copy()


    print(
        f"\nRestaurants after constraints: "
        f"{len(candidates):,}"
    )


    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    feature_rows = []
    restaurant_rows = []


    for _, restaurant in candidates.iterrows():

        feature_row = build_features(
            group,
            profile,
            restaurant
        )

        feature_rows.append(
            feature_row
        )

        restaurant_rows.append(
            restaurant
        )


    X = pd.DataFrame(
        feature_rows
    )

    X = X[
        feature_columns
    ]


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X
    )

    predictions = np.clip(
        predictions,
        0,
        1
    )


    # --------------------------------------------------------
    # Result dataframe
    # --------------------------------------------------------

    results = pd.DataFrame(
        restaurant_rows
    ).reset_index(drop=True)

    feature_df = pd.DataFrame(
        feature_rows
    ).reset_index(drop=True)


    results[
        "suitability_score"
    ] = predictions


    for column in [
        "cuisine_match",
        "budget_match",
        "dietary_compatibility",
        "restaurant_type_match",
        "location_match",
        "booking_match",
        "online_order_match"
    ]:

        results[column] = feature_df[
            column
        ]


    # --------------------------------------------------------
    # Dietary conflict detection
    # --------------------------------------------------------

    dietary_preferences = [
        normalize_text(value)
        for value in group[
            "dietary_preference"
        ]
    ]

    specific_dietary = set(
        value
        for value in dietary_preferences
        if value != "no preference"
    )

    dietary_conflict = (
        len(specific_dietary) > 1
    )


    results[
        "dietary_conflict"
    ] = dietary_conflict


    results[
        "booking_required"
    ] = profile[
        "booking_required"
    ]


    results[
        "online_preferred"
    ] = profile[
        "online_preferred"
    ]


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results = results.sort_values(
        "suitability_score",
        ascending=False
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # Diversity
    # --------------------------------------------------------

    top_results = apply_diversity(
        results,
        top_n
    ).copy()


    top_results[
        "rank"
    ] = range(
        1,
        len(top_results) + 1
    )


    # --------------------------------------------------------
    # Explanations
    # --------------------------------------------------------

    top_results[
        "explanation"
    ] = top_results.apply(
        generate_explanation,
        axis=1
    )


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        f"TOP {len(top_results)} RECOMMENDATIONS"
    )

    print("=" * 70)


    for _, row in top_results.iterrows():

        print(
            f"\n{int(row['rank'])}. "
            f"{row['name']}"
        )

        print(
            f"   Group Match: "
            f"{row['suitability_score'] * 100:.1f}%"
        )

        print(
            f"   Location: "
            f"{row['location']}"
        )

        print(
            f"   Cuisine: "
            f"{row['cuisines']}"
        )

        if pd.isna(
            row["cost_for_two"]
        ):

            print(
                "   Cost for two: "
                "Not available"
            )

        else:

            print(
                f"   Cost for two: "
                f"₹{row['cost_for_two']:.0f}"
            )


        if pd.isna(
            row["rating"]
        ):

            print(
                "   Rating: "
                "Not available"
            )

        else:

            print(
                f"   Rating: "
                f"{row['rating']:.1f}/5"
            )


        print(
            f"   Cuisine compatibility: "
            f"{row['cuisine_match'] * 100:.1f}%"
        )

        print(
            f"   Budget compatibility: "
            f"{row['budget_match'] * 100:.1f}%"
        )

        print(
            f"   Dietary compatibility: "
            f"{row['dietary_compatibility'] * 100:.1f}%"
        )

        print(
            f"   Location compatibility: "
            f"{row['location_match'] * 100:.1f}%"
        )

        print(
            f"   Explanation: "
            f"{row['explanation']}"
        )


    # --------------------------------------------------------
    # Save recommendations
    # --------------------------------------------------------

    output_columns = [
        "rank",
        "name",
        "location",
        "cuisines",
        "rest_type",
        "rating",
        "cost_for_two",
        "suitability_score",
        "cuisine_match",
        "budget_match",
        "dietary_compatibility",
        "restaurant_type_match",
        "location_match",
        "booking_match",
        "online_order_match",
        "online_order",
        "book_table",
        "explanation"
    ]


    recommendation_path = os.path.join(
        OUTPUT_DIR,
        f"{group_id}_recommendations.csv"
    )


    top_results[
        output_columns
    ].to_csv(
        recommendation_path,
        index=False
    )


    # --------------------------------------------------------
    # Save profile
    # --------------------------------------------------------

    profile_path = os.path.join(
        OUTPUT_DIR,
        f"{group_id}_profile.txt"
    )


    with open(
        profile_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"SmartDine Group Profile: "
            f"{group_id}\n"
        )

        f.write(
            "=" * 50 + "\n\n"
        )

        f.write(
            f"Group size: {len(group)}\n"
        )

        f.write(
            "Preferred cuisines: "
            + ", ".join(
                profile["cuisines"]
            )
            + "\n"
        )

        f.write(
            f"Weighted budget: "
            f"₹{profile['budget']:.0f}\n"
        )

        f.write(
            "Dietary preferences: "
            + ", ".join(
                profile["dietary"]
            )
            + "\n"
        )

        f.write(
            "Restaurant types: "
            + ", ".join(
                profile["restaurant_types"]
            )
            + "\n"
        )

        f.write(
            "Preferred locations: "
            + ", ".join(
                profile["locations"]
            )
            + "\n"
        )

        f.write(
            f"Group agreement: "
            f"{profile['group_agreement']:.3f}\n"
        )


    print(
        f"\nSaved recommendations: "
        f"{recommendation_path}"
    )

    print(
        f"Saved group profile: "
        f"{profile_path}"
    )


    return top_results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\nAvailable groups:")

    available_groups = (
        groups["group_id"]
        .drop_duplicates()
        .tolist()
    )

    print(
        ", ".join(
            available_groups[:20]
        )
    )

    if len(available_groups) > 20:

        print(
            f"... and "
            f"{len(available_groups) - 20} more"
        )


    # Test five different groups
    for group_id in [
        "G001",
        "G002",
        "G003",
        "G004",
        "G005"
    ]:
        recommend_for_group(
            group_id,
            top_n=TOP_N
        )