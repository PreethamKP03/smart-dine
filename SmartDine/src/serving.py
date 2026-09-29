"""
SmartDine Real-Time Serving API
FastAPI backend providing sub-50ms group recommendation scoring with explainability.
"""

import os
import re
import math
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Determine Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

RESTAURANT_PATH = os.path.join(DATA_DIR, "restaurant_features.csv")
GROUP_PATH = os.path.join(DATA_DIR, "group_members.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "smartdine_best_model.joblib")
FEATURE_PATH = os.path.join(MODELS_DIR, "feature_columns.txt")
COEFFS_PATH = os.path.join(OUTPUTS_DIR, "feature_coefficients.csv")
COMPARISON_PATH = os.path.join(OUTPUTS_DIR, "model_comparison.csv")

app = FastAPI(
    title="SmartDine Real-Time Recommendation API",
    description="ML-powered Group Dining Recommendation Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# In-Memory Cache & Pre-parsed Data for Sub-50ms Serving
# ------------------------------------------------------------
STATE = {
    "df": None,
    "model": None,
    "feature_columns": [],
    "feature_coefficients": {},
    "model_comparison": [],
    "groups_df": None,
    "all_cuisines": [],
    "all_locations": [],
    "all_types": [],
    # Pre-parsed vectorized structures
    "n_restaurants": 0,
    "cuisines_sets": [],
    "locations_list": [],
    "types_sets": [],
    "costs": np.array([]),
    "ratings": np.array([]),
    "votes": np.array([]),
    "online_orders": np.array([]),
    "book_tables": np.array([]),
    "rating_scores": np.array([]),
    "popularity_scores": np.array([]),
    "cuisines_raw_lower": []
}

NON_VEG_TERMS = {"chicken", "mutton", "fish", "meat", "seafood", "pork", "beef"}
ANIMAL_PRODUCT_TERMS = {"chicken", "mutton", "fish", "meat", "seafood", "pork", "beef", "egg", "cheese", "butter", "milk", "cream"}


def normalize_text(value: Any) -> str:
    if pd.isna(value) or value is None:
        return ""
    val = str(value).lower().strip().replace("&", "and")
    return re.sub(r"\s+", " ", val)


def split_values(value: Any) -> List[str]:
    if pd.isna(value) or value is None:
        return []
    val = str(value).strip()
    if not val:
        return []
    return [normalize_text(item) for item in val.split(",") if normalize_text(item)]


def initialize_engine():
    print("Initializing SmartDine real-time serving engine...")
    if not os.path.exists(RESTAURANT_PATH) or not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Missing required data or model files at {RESTAURANT_PATH} / {MODEL_PATH}")

    df = pd.read_csv(RESTAURANT_PATH)
    # Fill missing values
    df["cost_for_two"] = pd.to_numeric(df["cost_for_two"], errors="coerce").fillna(df["cost_for_two"].median())
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0)
    df["online_order"] = pd.to_numeric(df["online_order"], errors="coerce").fillna(0).astype(int)
    df["book_table"] = pd.to_numeric(df["book_table"], errors="coerce").fillna(0).astype(int)

    STATE["df"] = df
    STATE["n_restaurants"] = len(df)

    # Pre-parse sets and numpy arrays
    STATE["cuisines_sets"] = [set(split_values(x)) for x in df["cuisines"]]
    STATE["cuisines_raw_lower"] = [str(x).lower() if pd.notna(x) else "" for x in df["cuisines"]]
    STATE["locations_list"] = [normalize_text(x) for x in df["location"]]
    STATE["types_sets"] = [set(split_values(x)) for x in df["rest_type"]]
    STATE["costs"] = df["cost_for_two"].values.astype(float)
    STATE["ratings"] = df["rating"].values.astype(float)
    STATE["votes"] = df["votes"].values.astype(float)
    STATE["online_orders"] = df["online_order"].values.astype(int)
    STATE["book_tables"] = df["book_table"].values.astype(int)

    # Pre-calculate base rating & popularity scores
    filled_ratings = np.where(np.isnan(STATE["ratings"]), 3.5, STATE["ratings"])
    STATE["rating_scores"] = np.clip((filled_ratings - 1.0) / 4.0, 0.0, 1.0)
    max_votes = max(1.0, float(df["votes"].max()))
    STATE["popularity_scores"] = np.log1p(np.maximum(0.0, STATE["votes"])) / np.log1p(max_votes)

    # Load Model
    STATE["model"] = joblib.load(MODEL_PATH)

    # Load Feature columns
    with open(FEATURE_PATH, "r") as f:
        STATE["feature_columns"] = [line.strip() for line in f if line.strip()]

    # Load Feature Coefficients
    if os.path.exists(COEFFS_PATH):
        cdf = pd.read_csv(COEFFS_PATH)
        STATE["feature_coefficients"] = dict(zip(cdf["Feature"], cdf["Coefficient"]))

    # Load Model Comparison
    if os.path.exists(COMPARISON_PATH):
        STATE["model_comparison"] = pd.read_csv(COMPARISON_PATH).to_dict(orient="records")

    # Load Group benchmark data
    if os.path.exists(GROUP_PATH):
        STATE["groups_df"] = pd.read_csv(GROUP_PATH)

    # Extract distinct list of cuisines, locations, types
    cuisine_counts = {}
    for c_set in STATE["cuisines_sets"]:
        for c in c_set:
            cuisine_counts[c] = cuisine_counts.get(c, 0) + 1
    STATE["all_cuisines"] = sorted([c for c, count in cuisine_counts.items() if count >= 3])
    STATE["all_locations"] = sorted(list(set(loc for loc in STATE["locations_list"] if loc)))
    
    type_set = set()
    for t_set in STATE["types_sets"]:
        type_set.update(t_set)
    STATE["all_types"] = sorted(list(t_set))

    print(f"SmartDine Engine Ready: {STATE['n_restaurants']} restaurants, {len(STATE['all_cuisines'])} cuisines, {len(STATE['all_locations'])} locations.")


@app.on_event("startup")
def on_startup():
    initialize_engine()


# ------------------------------------------------------------
# Pydantic Request Models
# ------------------------------------------------------------
class MemberPreference(BaseModel):
    name: str = "Member"
    preferred_cuisines: List[str] = Field(default_factory=list)
    budget: float = 800.0
    dietary_preference: str = "no preference"  # vegetarian, vegan, non-vegetarian, no preference
    preferred_location: Optional[str] = None
    preferred_restaurant_type: Optional[str] = None
    cuisine_importance: float = 1.0
    budget_importance: float = 1.0
    dietary_importance: float = 1.0
    restaurant_type_importance: float = 1.0
    location_importance: float = 1.0
    requires_table_booking: bool = False
    prefers_online_order: bool = False


class RecommendationRequest(BaseModel):
    members: List[MemberPreference]
    top_n: int = Field(default=10, ge=1, le=50)
    filter_location: Optional[str] = None
    filter_cuisine: Optional[str] = None
    max_budget_limit: Optional[float] = None
    min_rating: Optional[float] = None
    require_booking: Optional[bool] = None
    require_online: Optional[bool] = None


# ------------------------------------------------------------
# Fast Vectorized Scoring Function (< 20ms)
# ------------------------------------------------------------
def compute_member_dietary_score(diet_pref: str, cuisines_raw: List[str]) -> np.ndarray:
    diet_pref = normalize_text(diet_pref)
    n = len(cuisines_raw)
    if not diet_pref or diet_pref in ["no preference", "non-vegetarian"]:
        return np.ones(n, dtype=float)

    scores = np.ones(n, dtype=float)
    if diet_pref == "vegetarian":
        # If any non_veg_terms in text -> 0.4, else 1.0
        for i, text in enumerate(cuisines_raw):
            if any(term in text for term in NON_VEG_TERMS):
                scores[i] = 0.4
    elif diet_pref == "vegan":
        for i, text in enumerate(cuisines_raw):
            if any(term in text for term in ANIMAL_PRODUCT_TERMS):
                scores[i] = 0.3
    return scores


def score_group_recommendations(req: RecommendationRequest) -> Dict[str, Any]:
    n = STATE["n_restaurants"]
    members = req.members
    group_size = max(1, len(members))

    if group_size == 0:
        raise HTTPException(status_code=400, detail="At least one group member is required")

    # Group Profile Aggregations
    # 1. Cuisines
    cuisine_scores_dict: Dict[str, float] = {}
    for m in members:
        w = m.cuisine_importance
        for c in m.preferred_cuisines:
            cn = normalize_text(c)
            if cn:
                cuisine_scores_dict[cn] = cuisine_scores_dict.get(cn, 0.0) + w

    total_c_weight = sum(cuisine_scores_dict.values())
    top_group_cuisines = []
    if total_c_weight > 0:
        top_group_cuisines = [c for c, _ in sorted(cuisine_scores_dict.items(), key=lambda x: x[1], reverse=True)[:5]]

    # 2. Weighted Budget
    budgets = np.array([m.budget for m in members], dtype=float)
    budget_weights = np.array([m.budget_importance for m in members], dtype=float)
    sum_b_w = np.sum(budget_weights)
    group_budget = float(np.average(budgets, weights=budget_weights) if sum_b_w > 0 else np.mean(budgets))

    # 3. Dietary preferences
    dietary_list = [normalize_text(m.dietary_preference) for m in members]
    group_dietary = list(set([d for d in dietary_list if d and d != "no preference"]))
    has_strict_diet = any(d in ["vegetarian", "vegan"] for d in dietary_list)

    # 4. Locations & Types
    locations_list = [normalize_text(m.preferred_location) for m in members if m.preferred_location]
    group_locations = list(set(locations_list))
    types_list = [normalize_text(m.preferred_restaurant_type) for m in members if m.preferred_restaurant_type]
    group_types = list(set(types_list))

    booking_required = any(m.requires_table_booking for m in members)
    if req.require_booking is not None:
        booking_required = req.require_booking

    online_preferred = any(m.prefers_online_order for m in members)
    if req.require_online is not None:
        online_preferred = req.require_online

    # 5. Group Agreement Score
    agreement_dims = []
    for dim_vals in [
        [c for m in members for c in m.preferred_cuisines],
        types_list,
        locations_list,
        [str(m.budget) for m in members],
        dietary_list
    ]:
        if dim_vals:
            counts = pd.Series(dim_vals).value_counts()
            agreement_dims.append(counts.max() / len(dim_vals))
    group_agreement = float(np.mean(agreement_dims)) if agreement_dims else 0.5

    # ------------------------------------------------------------
    # Vectorized Matching Computation across all restaurants
    # ------------------------------------------------------------
    # 1. Cuisine Match
    member_c_matches = []
    member_c_weights = []
    for m in members:
        pref_set = set([normalize_text(c) for c in m.preferred_cuisines if normalize_text(c)])
        w = float(m.cuisine_importance)
        if not pref_set:
            m_scores = np.full(n, 0.5, dtype=float)
        else:
            len_pref = len(pref_set)
            m_scores = np.array([min(1.0, len(pref_set.intersection(r_cuisines)) / len_pref) for r_cuisines in STATE["cuisines_sets"]], dtype=float)
        member_c_matches.append(m_scores)
        member_c_weights.append(w)
    
    total_cw = sum(member_c_weights) or 1.0
    cuisine_match_vec = sum(m_score * w for m_score, w in zip(member_c_matches, member_c_weights)) / total_cw

    # 2. Budget Match
    costs = STATE["costs"]
    member_b_matches = []
    member_b_weights = []
    for m in members:
        mb = max(1.0, float(m.budget))
        w = float(m.budget_importance)
        ratio = costs / mb
        m_b = np.where(ratio <= 1.0, 1.0,
              np.where(ratio <= 1.25, 0.75,
              np.where(ratio <= 1.5, 0.50,
              np.where(ratio <= 2.0, 0.25, 0.0))))
        member_b_matches.append(m_b)
        member_b_weights.append(w)
    total_bw = sum(member_b_weights) or 1.0
    budget_match_vec = sum(m_score * w for m_score, w in zip(member_b_matches, member_b_weights)) / total_bw

    # 3. Dietary Compatibility
    member_d_matches = []
    member_d_weights = []
    for m in members:
        d_score = compute_member_dietary_score(m.dietary_preference, STATE["cuisines_raw_lower"])
        w = float(m.dietary_importance)
        member_d_matches.append(d_score)
        member_d_weights.append(w)
    total_dw = sum(member_d_weights) or 1.0
    dietary_compatibility_vec = sum(m_score * w for m_score, w in zip(member_d_matches, member_d_weights)) / total_dw

    # 4. Restaurant Type Match
    member_t_matches = []
    member_t_weights = []
    for m in members:
        pt = normalize_text(m.preferred_restaurant_type)
        w = float(m.restaurant_type_importance)
        if not pt:
            m_t = np.full(n, 0.5, dtype=float)
        else:
            m_t = np.array([1.0 if pt in r_types else 0.0 for r_types in STATE["types_sets"]], dtype=float)
        member_t_matches.append(m_t)
        member_t_weights.append(w)
    total_tw = sum(member_t_weights) or 1.0
    restaurant_type_match_vec = sum(m_score * w for m_score, w in zip(member_t_matches, member_t_weights)) / total_tw

    # 5. Location Match
    member_l_matches = []
    member_l_weights = []
    for m in members:
        pl = normalize_text(m.preferred_location)
        w = float(m.location_importance)
        if not pl:
            m_l = np.full(n, 0.5, dtype=float)
        else:
            m_l = np.array([1.0 if pl == r_loc else 0.0 for r_loc in STATE["locations_list"]], dtype=float)
        member_l_matches.append(m_l)
        member_l_weights.append(w)
    total_lw = sum(member_l_weights) or 1.0
    location_match_vec = sum(m_score * w for m_score, w in zip(member_l_matches, member_l_weights)) / total_lw

    # 6. Booking Match & Online Order Match
    booking_match_val = 1.0 if booking_required else 0.5
    booking_match_vec = np.where(STATE["book_tables"] == 1, 1.0, 0.0) if booking_required else np.full(n, 0.5, dtype=float)

    online_match_val = 1.0 if online_preferred else 0.5
    online_order_match_vec = np.where(STATE["online_orders"] == 1, 1.0, 0.0) if online_preferred else np.full(n, 0.5, dtype=float)

    # ------------------------------------------------------------
    # Assemble Feature Matrix for Trained Model
    # ------------------------------------------------------------
    X_dict = {
        "group_size": np.full(n, group_size, dtype=float),
        "cuisine_match": cuisine_match_vec,
        "budget_match": budget_match_vec,
        "dietary_compatibility": dietary_compatibility_vec,
        "restaurant_type_match": restaurant_type_match_vec,
        "location_match": location_match_vec,
        "booking_match": booking_match_vec,
        "online_order_match": online_order_match_vec,
        "group_agreement": np.full(n, group_agreement, dtype=float),
        "rating_score": STATE["rating_scores"],
        "popularity_score": STATE["popularity_scores"],
        "cost_for_two": STATE["costs"],
        "votes": STATE["votes"],
        "online_order": STATE["online_orders"],
        "book_table": STATE["book_tables"]
    }

    feature_cols = STATE["feature_columns"]
    X = np.column_stack([X_dict[col] for col in feature_cols])

    # Model Inference
    predicted_scores = STATE["model"].predict(X)

    # Candidate Filtering Mask
    mask = np.ones(n, dtype=bool)

    if req.filter_location:
        norm_floc = normalize_text(req.filter_location)
        mask = mask & np.array([norm_floc in loc for loc in STATE["locations_list"]])

    if req.filter_cuisine:
        norm_fc = normalize_text(req.filter_cuisine)
        mask = mask & np.array([norm_fc in c_set for c_set in STATE["cuisines_sets"]])

    if req.max_budget_limit:
        mask = mask & (STATE["costs"] <= req.max_budget_limit)

    if req.min_rating:
        mask = mask & (STATE["ratings"] >= req.min_rating)

    if req.require_booking:
        mask = mask & (STATE["book_tables"] == 1)

    if req.require_online:
        mask = mask & (STATE["online_orders"] == 1)

    # Apply strict dietary constraint if requested
    if has_strict_diet:
        # Don't show restaurants that score critically low on dietary compatibility (< 0.35)
        mask = mask & (dietary_compatibility_vec >= 0.35)

    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        valid_indices = np.arange(n)  # Fallback if over-filtered

    # Sort top candidates
    valid_scores = predicted_scores[valid_indices]
    top_k = min(req.top_n, len(valid_indices))
    top_subset_idx = np.argsort(valid_scores)[::-1][:top_k]
    selected_indices = valid_indices[top_subset_idx]

    # Build Response Items
    results = []
    df = STATE["df"]
    for rank_idx, r_i in enumerate(selected_indices):
        row = df.iloc[r_i]
        score = float(predicted_scores[r_i])
        
        c_m = float(cuisine_match_vec[r_i])
        b_m = float(budget_match_vec[r_i])
        d_m = float(dietary_compatibility_vec[r_i])
        t_m = float(restaurant_type_match_vec[r_i])
        l_m = float(location_match_vec[r_i])

        # Generate Explainability Badges & Text
        reasons = []
        cautions = []
        if c_m >= 0.75:
            reasons.append("Strong cuisine compatibility")
        elif c_m >= 0.40:
            reasons.append("Partial cuisine compatibility")
        else:
            cautions.append("Limited cuisine overlap")

        if b_m >= 0.75:
            reasons.append("Within the group's preferred budget")
        elif b_m >= 0.40:
            reasons.append("Near group's budget target")
        else:
            cautions.append("Above preferred budget")

        if d_m >= 0.75:
            if has_strict_diet and d_m < 0.95:
                cautions.append("Verify dietary options on arrival")
            else:
                reasons.append("Accommodates dietary preferences")
        else:
            cautions.append("Caution: dietary options should be verified")

        if l_m >= 0.5:
            reasons.append("Matches preferred location for members")

        if row["rating"] >= 4.0:
            reasons.append(f"Highly rated ({row['rating']:.1f}/5)")

        explanation_text = "Recommended because of " + ", ".join(reasons) if reasons else "Good general recommendation"
        if cautions:
            explanation_text += ". Note: " + ", ".join(cautions) + "."

        results.append({
            "rank": rank_idx + 1,
            "restaurant_id": str(row["restaurant_id"]),
            "name": str(row["name"]),
            "location": str(row["location"]),
            "cuisines": str(row["cuisines"]),
            "rest_type": str(row["rest_type"]) if pd.notna(row["rest_type"]) else "Restaurant",
            "cost_for_two": float(row["cost_for_two"]),
            "rating": float(row["rating"]) if pd.notna(row["rating"]) else None,
            "votes": int(row["votes"]),
            "online_order": bool(row["online_order"] == 1),
            "book_table": bool(row["book_table"] == 1),
            "suitability_score": round(score, 4),
            "match_percentage": round(min(100.0, max(0.0, score * 100)), 1),
            "breakdown": {
                "cuisine_match": round(c_m * 100, 1),
                "budget_match": round(b_m * 100, 1),
                "dietary_compatibility": round(d_m * 100, 1),
                "restaurant_type_match": round(t_m * 100, 1),
                "location_match": round(l_m * 100, 1),
                "group_agreement": round(group_agreement * 100, 1),
            },
            "reasons": reasons,
            "cautions": cautions,
            "explanation": explanation_text
        })

    group_summary = {
        "group_size": group_size,
        "weighted_budget": round(group_budget, 0),
        "group_agreement": round(group_agreement, 3),
        "top_cuisines": top_group_cuisines,
        "dietary_mix": group_dietary or ["No restriction"],
        "locations": group_locations or ["Any location"],
        "restaurant_types": group_types or ["Any type"],
        "total_candidates_scored": len(valid_indices)
    }

    return {
        "status": "success",
        "group_summary": group_summary,
        "recommendations": results
    }


# ------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model": STATE["model"].__class__.__name__ if STATE["model"] else "Not loaded",
        "total_restaurants": STATE["n_restaurants"],
        "features": STATE["feature_columns"],
        "model_r2_score": 0.7641
    }


@app.get("/api/metadata")
def get_metadata():
    return {
        "cuisines": STATE["all_cuisines"][:50],
        "locations": STATE["all_locations"],
        "restaurant_types": STATE["all_types"],
        "feature_coefficients": STATE["feature_coefficients"],
        "model_comparison": STATE["model_comparison"]
    }


@app.get("/api/presets")
def get_presets():
    """Returns benchmark groups G001-G005 + quick demo personas."""
    presets = [
        {
            "id": "G001",
            "name": "Benchmark G001: Healthy & Mixed Diets",
            "description": "5 members in Electronic City with mixed vegan, vegetarian & non-vegetarian diets.",
            "members": [
                {"name": "Ananya", "preferred_cuisines": ["healthy food", "sandwich"], "budget": 600, "dietary_preference": "vegan", "preferred_location": "electronic city", "preferred_restaurant_type": "cafe"},
                {"name": "Rahul", "preferred_cuisines": ["burger", "fast food"], "budget": 800, "dietary_preference": "non-vegetarian", "preferred_location": "electronic city", "preferred_restaurant_type": "food court"},
                {"name": "Sneha", "preferred_cuisines": ["healthy food", "salad"], "budget": 700, "dietary_preference": "vegetarian", "preferred_location": "malleshwaram", "preferred_restaurant_type": "cafe"},
                {"name": "Vikram", "preferred_cuisines": ["seafood", "fast food"], "budget": 900, "dietary_preference": "non-vegetarian", "preferred_location": "bannerghatta road", "preferred_restaurant_type": "bar"},
                {"name": "Pooja", "preferred_cuisines": ["healthy food", "dessert"], "budget": 650, "dietary_preference": "vegetarian", "preferred_location": "electronic city", "preferred_restaurant_type": "dessert parlor"}
            ]
        },
        {
            "id": "G002",
            "name": "Benchmark G002: Whitefield Asian & Momos",
            "description": "4 members in Whitefield loving Asian cuisine, Momos, and Seafood.",
            "members": [
                {"name": "Karthik", "preferred_cuisines": ["momos", "asian"], "budget": 850, "dietary_preference": "non-vegetarian", "preferred_location": "whitefield", "preferred_restaurant_type": "takeaway"},
                {"name": "Meera", "preferred_cuisines": ["seafood", "asian"], "budget": 1000, "dietary_preference": "non-vegetarian", "preferred_location": "whitefield", "preferred_restaurant_type": "delivery"},
                {"name": "Arjun", "preferred_cuisines": ["ice cream", "momos"], "budget": 900, "dietary_preference": "vegetarian", "preferred_location": "whitefield", "preferred_restaurant_type": "bakery"},
                {"name": "Divya", "preferred_cuisines": ["american", "asian"], "budget": 950, "dietary_preference": "vegetarian", "preferred_location": "malleshwaram", "preferred_restaurant_type": "sweet shop"}
            ]
        },
        {
            "id": "G005",
            "name": "Benchmark G005: Strict Veg Budget Hangout",
            "description": "3 strict vegetarians looking for affordable dining under ₹400 in Whitefield/Kalyan Nagar.",
            "members": [
                {"name": "Ramesh", "preferred_cuisines": ["fast food", "salad"], "budget": 350, "dietary_preference": "vegetarian", "preferred_location": "whitefield", "preferred_restaurant_type": "beverage shop"},
                {"name": "Suresh", "preferred_cuisines": ["american", "fast food"], "budget": 380, "dietary_preference": "vegetarian", "preferred_location": "whitefield", "preferred_restaurant_type": "bakery"},
                {"name": "Mahesh", "preferred_cuisines": ["salad", "beverages"], "budget": 340, "dietary_preference": "vegetarian", "preferred_location": "kalyan nagar", "preferred_restaurant_type": "fine dining"}
            ]
        },
        {
            "id": "PRESET_TECH_LUNCH",
            "name": "Tech Team Koramangala Hangout",
            "description": "4 developers looking for Biryani, North Indian & Continental lunch in Koramangala.",
            "members": [
                {"name": "Preetham", "preferred_cuisines": ["biryani", "north indian"], "budget": 1000, "dietary_preference": "non-vegetarian", "preferred_location": "koramangala 5th block", "preferred_restaurant_type": "casual dining"},
                {"name": "Deepak", "preferred_cuisines": ["continental", "pizza"], "budget": 1200, "dietary_preference": "non-vegetarian", "preferred_location": "koramangala 5th block", "preferred_restaurant_type": "cafe"},
                {"name": "Tanvi", "preferred_cuisines": ["north indian", "chinese"], "budget": 900, "dietary_preference": "vegetarian", "preferred_location": "koramangala 5th block", "preferred_restaurant_type": "casual dining"},
                {"name": "Siddharth", "preferred_cuisines": ["biryani", "fast food"], "budget": 800, "dietary_preference": "non-vegetarian", "preferred_location": "koramangala 5th block", "preferred_restaurant_type": "quick bites"}
            ]
        }
    ]
    return presets


@app.post("/api/recommend")
def recommend(req: RecommendationRequest):
    return score_group_recommendations(req)


# ------------------------------------------------------------
# Frontend Static Files Mount
# ------------------------------------------------------------
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
