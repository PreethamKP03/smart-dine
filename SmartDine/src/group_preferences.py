"""
SmartDine — Group Preference Aggregation Module
src/group_preferences.py

Responsibilities:
1. Model individual user dining preferences.
2. Aggregate multiple individual preferences into a cohesive Group Profile.
3. Compute group agreement / consensus metrics across dimensions.
4. Support context attributes: time of day, day type, group size.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional


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


class MemberPreference:
    def __init__(
        self,
        name: str = "Member",
        preferred_cuisines: Optional[List[str]] = None,
        budget: float = 800.0,
        dietary_preference: str = "no preference",
        preferred_location: str = "",
        min_rating: float = 3.5,
        preferred_restaurant_type: str = "",
        cuisine_importance: float = 1.0,
        budget_importance: float = 1.0,
        dietary_importance: float = 1.0,
        requires_booking: bool = False,
        prefers_online: bool = False
    ):
        self.name = name
        self.preferred_cuisines = [normalize_text(c) for c in (preferred_cuisines or []) if normalize_text(c)]
        self.budget = float(budget)
        self.dietary_preference = normalize_text(dietary_preference) or "no preference"
        self.preferred_location = normalize_text(preferred_location)
        self.min_rating = float(min_rating)
        self.preferred_restaurant_type = normalize_text(preferred_restaurant_type)
        self.cuisine_importance = float(cuisine_importance)
        self.budget_importance = float(budget_importance)
        self.dietary_importance = float(dietary_importance)
        self.requires_booking = bool(requires_booking)
        self.prefers_online = bool(prefers_online)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "preferred_cuisines": self.preferred_cuisines,
            "budget": self.budget,
            "dietary_preference": self.dietary_preference,
            "preferred_location": self.preferred_location,
            "min_rating": self.min_rating,
            "preferred_restaurant_type": self.preferred_restaurant_type,
            "cuisine_importance": self.cuisine_importance,
            "budget_importance": self.budget_importance,
            "dietary_importance": self.dietary_importance,
            "requires_booking": self.requires_booking,
            "prefers_online": self.prefers_online
        }


class GroupPreferences:
    """
    Combines individual member preferences into a structured group profile
    using weighted preference aggregation and consensus metrics.
    """

    def __init__(
        self,
        members: List[MemberPreference],
        time_of_day: str = "dinner",   # lunch, dinner, late night
        day_type: str = "weekend"       # weekday, weekend
    ):
        if not members:
            raise ValueError("Group must contain at least one member.")
        self.members = members
        self.time_of_day = normalize_text(time_of_day) or "dinner"
        self.day_type = normalize_text(day_type) or "weekend"
        self.group_size = len(members)

    def compute_group_profile(self) -> Dict[str, Any]:
        # 1. Aggregate Cuisines with Frequency & Weighting
        cuisine_scores: Dict[str, float] = {}
        for m in self.members:
            w = m.cuisine_importance
            for c in m.preferred_cuisines:
                cuisine_scores[c] = cuisine_scores.get(c, 0.0) + w

        total_cw = sum(cuisine_scores.values())
        top_cuisines = []
        if total_cw > 0:
            top_cuisines = [
                c for c, _ in sorted(cuisine_scores.items(), key=lambda x: x[1], reverse=True)[:6]
            ]

        # 2. Weighted Budget Calculation (avoiding simple blind average)
        budgets = np.array([m.budget for m in self.members], dtype=float)
        weights = np.array([m.budget_importance for m in self.members], dtype=float)
        sum_w = np.sum(weights)
        weighted_budget = float(np.average(budgets, weights=weights) if sum_w > 0 else np.mean(budgets))

        # 3. Minimum Rating (group requires the highest agreed standards)
        min_ratings = [m.min_rating for m in self.members]
        group_min_rating = float(np.percentile(min_ratings, 75)) if min_ratings else 3.5

        # 4. Dietary Reconciliations (strict safety first)
        dietary_set = set(m.dietary_preference for m in self.members if m.dietary_preference != "no preference")
        has_vegan = "vegan" in dietary_set
        has_veg = "vegetarian" in dietary_set or has_vegan

        # 5. Preferred Locations
        locations = [m.preferred_location for m in self.members if m.preferred_location]
        top_locations = list(set(locations))

        # 6. Restaurant Types
        types = [m.preferred_restaurant_type for m in self.members if m.preferred_restaurant_type]
        top_types = list(set(types))

        # 7. Consensus / Group Agreement Score
        agreement_dims = []
        for dim_vals in [
            [c for m in self.members for c in m.preferred_cuisines],
            types,
            locations,
            [str(m.budget) for m in self.members],
            [m.dietary_preference for m in self.members]
        ]:
            if dim_vals:
                counts = pd.Series(dim_vals).value_counts()
                agreement_dims.append(counts.max() / len(dim_vals))
        group_agreement = float(np.mean(agreement_dims)) if agreement_dims else 0.5

        return {
            "group_size": self.group_size,
            "weighted_budget": round(weighted_budget, 1),
            "min_rating": round(group_min_rating, 1),
            "top_cuisines": top_cuisines,
            "dietary_mix": list(dietary_set) or ["no preference"],
            "has_vegetarian": has_veg,
            "has_vegan": has_vegan,
            "preferred_locations": top_locations,
            "preferred_types": top_types,
            "group_agreement": round(group_agreement, 3),
            "time_of_day": self.time_of_day,
            "day_type": self.day_type,
            "booking_required": any(m.requires_booking for m in self.members),
            "online_preferred": any(m.prefers_online for m in self.members)
        }
