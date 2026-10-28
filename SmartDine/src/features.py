"""Shared training/inference features. No learned catalogue-wide imputation.

Unknown numeric values use explicit fixed neutral values only in model inputs.
Original unknown fields remain unknown in results and fail hard filters.
"""
import numpy as np
import pandas as pd
from .group_preferences import MemberPreference, GroupPreferences, normalize_text, split_values
from .schema import FEATURES

NONVEG = {'seafood', 'biryani', 'kebab', 'kabab', 'meat', 'steak', 'fish', 'chicken', 'mughlai'}


def members_from_frame(frame):
    result = []
    for row in frame.fillna('').to_dict('records'):
        args = {k: row[k] for k in ['budget', 'dietary_preference', 'preferred_location',
                'preferred_restaurant_type', 'cuisine_importance', 'budget_importance',
                'dietary_importance', 'restaurant_type_importance', 'location_importance'] if k in row}
        result.append(MemberPreference(name=row.get('member_id', 'Member'),
                      preferred_cuisines=split_values(row.get('preferred_cuisines', '')),
                      requires_booking=bool(row.get('requires_table_booking', 0)),
                      prefers_online=bool(row.get('prefers_online_order', 0)), **args))
    return result


def numeric(frame, key):
    return pd.to_numeric(frame.get(key, pd.Series(np.nan, index=frame.index)), errors='coerce').replace([np.inf, -np.inf], np.nan)


class RestaurantFeatureBuilder:
    def __init__(self, frame):
        required = {'restaurant_id', 'name'}
        if not required <= set(frame):
            raise ValueError('Restaurant data requires restaurant_id and name columns.')
        if frame.empty or frame.restaurant_id.isna().any() or frame.restaurant_id.duplicated().any():
            raise ValueError('Restaurant IDs must be nonempty and unique; catalogue cannot be empty.')
        self.df = frame.copy().reset_index(drop=True)
        for key in ['cuisines', 'location', 'rest_type']:
            if key not in self.df:
                self.df[key] = None
        for key in ['cost_for_two', 'rating', 'votes', 'online_order', 'book_table']:
            self.df[key] = numeric(self.df, key)
        self.df.loc[self.df.cost_for_two <= 0, 'cost_for_two'] = np.nan
        self.df.loc[~self.df.rating.between(0, 5), 'rating'] = np.nan
        self.df.loc[self.df.votes < 0, 'votes'] = np.nan
        for key in ['online_order', 'book_table']:
            self.df.loc[~self.df[key].isin([0, 1]), key] = np.nan
        self.cuisines = [set(split_values(x)) for x in self.df.cuisines]
        self.types = [set(split_values(x)) for x in self.df.rest_type]
        self.locations = [normalize_text(x) for x in self.df.location]
        self.nonveg = np.array([any(k in ' '.join(c) for k in NONVEG) for c in self.cuisines])

    def build(self, group):
        n = len(self.df)
        costs = self.df.cost_for_two.to_numpy(dtype=float)
        def average(values, attr):
            weights = np.array([getattr(m, attr) for m in group.members])
            return np.average(values, axis=0, weights=weights if weights.sum() else None)
        cuisine, budget, diet, types, locations, booking, online = [], [], [], [], [], [], []
        for m in group.members:
            pref = set(m.preferred_cuisines)
            cuisine.append(np.array([len(pref & c) / len(pref) if pref else .5 for c in self.cuisines]))
            ratio = costs / m.budget
            budget.append(np.where(np.isnan(ratio), .5, np.select(
                [ratio <= 1, ratio <= 1.25, ratio <= 1.5, ratio <= 2], [1, .75, .5, .25], 0)))
            # Deliberately weak heuristic, NEVER a verified dietary fact.
            if m.dietary_preference in {'vegetarian', 'vegan'}:
                d = np.where(self.nonveg, .25 if m.dietary_preference == 'vegan' else .5, .5)
            else:
                d = np.ones(n)
            diet.append(d)
            types.append(np.array([1. if m.preferred_restaurant_type in t else 0.
                                  for t in self.types]) if m.preferred_restaurant_type else np.full(n, .5))
            locations.append(np.array([float(m.preferred_location == l) for l in self.locations])
                             if m.preferred_location else np.full(n, .5))
            booking.append(self.df.book_table.fillna(.5).to_numpy() if m.requires_booking else np.ones(n))
            online.append(self.df.online_order.fillna(.5).to_numpy() if m.prefers_online else np.ones(n))
        out = pd.DataFrame({
            'group_size': np.full(n, group.group_size),
            'cuisine_match': average(cuisine, 'cuisine_importance'),
            'budget_match': average(budget, 'budget_importance'),
            'dietary_compatibility': average(diet, 'dietary_importance'),
            'restaurant_type_match': average(types, 'restaurant_type_importance'),
            'location_match': average(locations, 'location_importance'),
            'booking_match': np.mean(booking, axis=0), 'online_order_match': np.mean(online, axis=0),
            'group_agreement': np.full(n, group.compute_group_profile()['group_agreement']),
            'rating_score': self.df.rating.fillna(3.5).to_numpy() / 5,
            # Fixed reference 20,000 votes avoids fitting on test catalogue statistics.
            'popularity_score': np.clip(np.log1p(self.df.votes.fillna(0)) / np.log1p(20000), 0, 1),
            'cost_for_two': self.df.cost_for_two.fillna(800), 'votes': self.df.votes.fillna(0),
            'online_order': self.df.online_order.fillna(.5), 'book_table': self.df.book_table.fillna(.5)})
        return out[FEATURES].astype(float)
