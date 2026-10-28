from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from SmartDine.src.schema import FEATURES
from SmartDine.src.recommendation import SmartDineRecommender


@pytest.fixture
def restaurants():
    # Deliberately tiny fixtures, never passed off as the original dataset.
    return pd.DataFrame([
        dict(restaurant_id='b', name='Bistro', cuisines='Italian', location='North', rest_type='Cafe',
             rating=4.5, cost_for_two=900, votes=20, book_table=1, online_order=0),
        dict(restaurant_id='a', name='Budget', cuisines='Chinese', location='South', rest_type='Quick Bites',
             rating=3.8, cost_for_two=200, votes=5, book_table=0, online_order=1),
        dict(restaurant_id='c', name='Unknown', cuisines=None, location=None, rest_type=None,
             rating=None, cost_for_two=None, votes=None, book_table=None, online_order=None)])


@pytest.fixture
def engine(tmp_path, restaurants):
    data, model = tmp_path / 'restaurants.csv', tmp_path / 'model.joblib'
    restaurants.to_csv(data, index=False)
    rng = np.random.RandomState(7)
    x = pd.DataFrame(rng.uniform(size=(80, len(FEATURES))), columns=FEATURES)
    y = .6*x.cuisine_match + .3*x.budget_match + .1*x.rating_score
    joblib.dump(LinearRegression().fit(x, y), model)
    return SmartDineRecommender(data, model)
