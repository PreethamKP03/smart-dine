"""
SmartDine — Recommendation Engine Module
src/recommendation.py

Responsibilities:
1. Load cleaned restaurant dataset & trained ML model.
2. Score candidate restaurants against a group profile.
3. Compute context-aware features (time of day, group size, location, budget).
4. Predict compatibility scores using trained Linear Regression champion model.
5. Return ranked Top-N recommendations with natural-language explainable rationales.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

try:
    from src.group_preferences import GroupPreferences, MemberPreference, normalize_text, split_values
except ImportError:
    from group_preferences import GroupPreferences, MemberPreference, normalize_text, split_values


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "restaurant_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "smartdine_best_model.joblib")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_columns.txt")

NON_VEG_TERMS = {"chicken", "mutton", "fish", "meat", "seafood", "pork", "beef"}
ANIMAL_TERMS = {"chicken", "mutton", "fish", "meat", "seafood", "pork", "beef", "egg", "cheese", "butter", "milk", "cream"}


class SmartDineRecommender:
    def __init__(self, data_path=None, model_path=None):
        self.data_path = data_path or DATA_PATH
        self.model_path = model_path or MODEL_PATH
        self.df = None
        self.model = None
        self.feature_columns = []
        self._load_resources()

    def _load_resources(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Restaurant features dataset not found at: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df["cost_for_two"] = pd.to_numeric(self.df["cost_for_two"], errors="coerce").fillna(self.df["cost_for_two"].median())
        self.df["rating"] = pd.to_numeric(self.df["rating"], errors="coerce")
        self.df["votes"] = pd.to_numeric(self.df["votes"], errors="coerce").fillna(0)
        self.df["online_order"] = pd.to_numeric(self.df["online_order"], errors="coerce").fillna(0).astype(int)
        self.df["book_table"] = pd.to_numeric(self.df["book_table"], errors="coerce").fillna(0).astype(int)

        # Pre-parse sets and numpy arrays for fast matching
        self.cuisines_sets = [set(split_values(x)) for x in self.df["cuisines"]]
        self.cuisines_raw_lower = [str(x).lower() if pd.notna(x) else "" for x in self.df["cuisines"]]
        self.locations_list = [normalize_text(x) for x in self.df["location"]]
        self.types_sets = [set(split_values(x)) for x in self.df["rest_type"]]
        self.costs = self.df["cost_for_two"].values.astype(float)
        self.ratings = self.df["rating"].values.astype(float)
        self.votes = self.df["votes"].values.astype(float)

        filled_ratings = np.where(np.isnan(self.ratings), 3.5, self.ratings)
        self.rating_scores = np.clip((filled_ratings - 1.0) / 4.0, 0.0, 1.0)
        max_votes = max(1.0, float(self.df["votes"].max()))
        self.popularity_scores = np.log1p(np.maximum(0.0, self.votes)) / np.log1p(max_votes)

        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        if os.path.exists(FEATURE_PATH):
            with open(FEATURE_PATH, "r") as f:
                self.feature_columns = [l.strip() for l in f if l.strip()]

    def recommend(
        self,
        group_prefs: GroupPreferences,
        top_n: int = 10,
        filter_location: Optional[str] = None,
        filter_cuisine: Optional[str] = None,
        max_budget: Optional[float] = None,
        min_rating: Optional[float] = None,
        require_booking: Optional[bool] = None,
        require_online: Optional[bool] = None
    ) -> Dict[str, Any]:
        profile = group_prefs.compute_group_profile()
        members = group_prefs.members
        n = len(self.df)
        group_size = len(members)

        # 1. Member Cuisine Matching
        m_c_matches = []
        for m in members:
            pref = set(m.preferred_cuisines)
            if not pref:
                m_c_matches.append(np.full(n, 0.5, dtype=float))
            else:
                lp = len(pref)
                m_c_matches.append(np.array([min(1.0, len(pref.intersection(r)) / lp) for r in self.cuisines_sets]))
        c_weights = [m.cuisine_importance for m in members]
        cuisine_match_vec = sum(s * w for s, w in zip(m_c_matches, c_weights)) / (sum(c_weights) or 1.0)

        # 2. Member Budget Matching
        m_b_matches = []
        for m in members:
            ratio = self.costs / max(1.0, m.budget)
            m_b = np.where(ratio <= 1.0, 1.0,
                  np.where(ratio <= 1.25, 0.75,
                  np.where(ratio <= 1.5, 0.50,
                  np.where(ratio <= 2.0, 0.25, 0.0))))
            m_b_matches.append(m_b)
        b_weights = [m.budget_importance for m in members]
        budget_match_vec = sum(s * w for s, w in zip(m_b_matches, b_weights)) / (sum(b_weights) or 1.0)

        # 3. Dietary Compatibility
        m_d_matches = []
        for m in members:
            d = m.dietary_preference
            if d in ["no preference", "non-vegetarian"] or not d:
                m_d_matches.append(np.ones(n, dtype=float))
            elif d == "vegetarian":
                scores = np.ones(n, dtype=float)
                for i, text in enumerate(self.cuisines_raw_lower):
                    if any(t in text for t in NON_VEG_TERMS):
                        scores[i] = 0.4
                m_d_matches.append(scores)
            elif d == "vegan":
                scores = np.ones(n, dtype=float)
                for i, text in enumerate(self.cuisines_raw_lower):
                    if any(t in text for t in ANIMAL_TERMS):
                        scores[i] = 0.3
                m_d_matches.append(scores)
            else:
                m_d_matches.append(np.full(n, 0.5, dtype=float))
        d_weights = [m.dietary_importance for m in members]
        dietary_compatibility_vec = sum(s * w for s, w in zip(m_d_matches, d_weights)) / (sum(d_weights) or 1.0)

        # 4. Location Match
        m_l_matches = []
        for m in members:
            pl = m.preferred_location
            if not pl:
                m_l_matches.append(np.full(n, 0.5, dtype=float))
            else:
                m_l_matches.append(np.array([1.0 if pl == loc else 0.0 for loc in self.locations_list]))
        location_match_vec = sum(m_l_matches) / len(m_l_matches)

        # 5. Restaurant Type Match
        m_t_matches = []
        for m in members:
            pt = m.preferred_restaurant_type
            if not pt:
                m_t_matches.append(np.full(n, 0.5, dtype=float))
            else:
                m_t_matches.append(np.array([1.0 if pt in types else 0.0 for types in self.types_sets]))
        restaurant_type_match_vec = sum(m_t_matches) / len(m_t_matches)

        # 6. Booking & Online Order Match
        booking_match_vec = np.where(self.df["book_table"].values == 1, 1.0, 0.0) if profile["booking_required"] else np.full(n, 0.5)
        online_order_match_vec = np.where(self.df["online_order"].values == 1, 1.0, 0.0) if profile["online_preferred"] else np.full(n, 0.5)

        # 7. Assemble Features and Predict
        X_dict = {
            "group_size": np.full(n, group_size, dtype=float),
            "cuisine_match": cuisine_match_vec,
            "budget_match": budget_match_vec,
            "dietary_compatibility": dietary_compatibility_vec,
            "restaurant_type_match": restaurant_type_match_vec,
            "location_match": location_match_vec,
            "booking_match": booking_match_vec,
            "online_order_match": online_order_match_vec,
            "group_agreement": np.full(n, profile["group_agreement"], dtype=float),
            "rating_score": self.rating_scores,
            "popularity_score": self.popularity_scores,
            "cost_for_two": self.costs,
            "votes": self.votes,
            "online_order": self.df["online_order"].values,
            "book_table": self.df["book_table"].values
        }

        X = np.column_stack([X_dict[col] for col in self.feature_columns])
        scores = self.model.predict(X)

        # 8. Filter Candidates
        mask = np.ones(n, dtype=bool)
        if filter_location:
            floc = normalize_text(filter_location)
            mask &= np.array([floc in loc for loc in self.locations_list])
        if filter_cuisine:
            fcui = normalize_text(filter_cuisine)
            mask &= np.array([fcui in s for s in self.cuisines_sets])
        if max_budget:
            mask &= (self.costs <= max_budget)
        if min_rating:
            mask &= (self.ratings >= min_rating)
        if require_booking:
            mask &= (self.df["book_table"].values == 1)
        if require_online:
            mask &= (self.df["online_order"].values == 1)

        # Strict dietary protection
        if profile["has_vegetarian"] or profile["has_vegan"]:
            mask &= (dietary_compatibility_vec >= 0.35)

        valid_idx = np.where(mask)[0]
        if len(valid_idx) == 0:
            valid_idx = np.arange(n)

        # Rank Top-N
        valid_scores = scores[valid_idx]
        k = min(top_n, len(valid_idx))
        top_k_indices = valid_idx[np.argsort(valid_scores)[::-1][:k]]

        recommendations = []
        for rank, idx in enumerate(top_k_indices, 1):
            row = self.df.iloc[idx]
            s = float(scores[idx])
            cm = float(cuisine_match_vec[idx])
            bm = float(budget_match_vec[idx])
            dm = float(dietary_compatibility_vec[idx])
            lm = float(location_match_vec[idx])

            reasons = []
            cautions = []
            if cm >= 0.75:
                reasons.append("Strong cuisine compatibility")
            elif cm >= 0.40:
                reasons.append("Partial cuisine match")

            if bm >= 0.75:
                reasons.append("Within group's preferred budget")
            elif bm < 0.40:
                cautions.append("Above target budget")

            if dm >= 0.75:
                reasons.append("Accommodates dietary preferences")
            else:
                cautions.append("Verify dietary options on arrival")

            if lm >= 0.5:
                reasons.append("Preferred location for group members")

            if row["rating"] >= 4.0:
                reasons.append(f"Highly rated ({row['rating']:.1f}/5)")

            explanation = "Recommended because of " + ", ".join(reasons) if reasons else "Good general recommendation"
            if cautions:
                explanation += ". Note: " + ", ".join(cautions) + "."

            recommendations.append({
                "rank": rank,
                "name": str(row["name"]),
                "location": str(row["location"]),
                "cuisines": str(row["cuisines"]),
                "rest_type": str(row["rest_type"]) if pd.notna(row["rest_type"]) else "Restaurant",
                "cost_for_two": float(row["cost_for_two"]),
                "rating": float(row["rating"]) if pd.notna(row["rating"]) else None,
                "votes": int(row["votes"]),
                "online_order": bool(row["online_order"] == 1),
                "book_table": bool(row["book_table"] == 1),
                "compatibility_score": round(s, 4),
                "match_percentage": round(min(100.0, max(0.0, s * 100)), 1),
                "breakdown": {
                    "cuisine_match": round(cm * 100, 1),
                    "budget_match": round(bm * 100, 1),
                    "dietary_compatibility": round(dm * 100, 1),
                    "location_match": round(lm * 100, 1)
                },
                "explanation": explanation
            })

        return {
            "group_profile": profile,
            "total_candidates_evaluated": len(valid_idx),
            "recommendations": recommendations
        }


if __name__ == "__main__":
    recommender = SmartDineRecommender()
    members = [
        MemberPreference(name="Ananya", preferred_cuisines=["healthy food", "sandwich"], budget=600, dietary_preference="vegan", preferred_location="electronic city"),
        MemberPreference(name="Rahul", preferred_cuisines=["burger", "fast food"], budget=800, dietary_preference="non-vegetarian", preferred_location="electronic city")
    ]
    group = GroupPreferences(members=members, time_of_day="lunch", day_type="weekend")
    result = recommender.recommend(group, top_n=5)
    print("Recommendations complete! Top pick:", result["recommendations"][0]["name"])
