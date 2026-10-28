"""Replay the supplied legacy feature/target formulas without modifying active artifacts.

Reads the preserved snapshot. This is forensic evidence, not a supported model
deployment across scikit-learn versions. Predicts the saved linear equation
numerically after extracting its coefficients, and records the version warning.
"""
from pathlib import Path
import io
import sys
import warnings
import zipfile
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from SmartDine.src.schema import FEATURES, TARGET_WEIGHTS, write_json, sha256
from SmartDine.src.train_model import metrics


def replay():
    backup = ROOT / '.local-backup' / 'original-snapshot.zip'
    with zipfile.ZipFile(backup) as z:
        restaurants = pd.read_csv(io.BytesIO(z.read('SmartDine/data/processed/restaurant_features.csv')))
        groups = pd.read_csv(io.BytesIO(z.read('SmartDine/data/processed/group_members.csv')))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            model = joblib.load(io.BytesIO(z.read('SmartDine/models/smartdine_best_model.joblib')))
    cuisines = [set(str(x).strip().lower().split(', ')) if pd.notna(x) else set() for x in restaurants.cuisines]
    texts = [' '.join(c) for c in cuisines]
    types = [str(x).lower().strip() if pd.notna(x) else '' for x in restaurants.rest_type]
    locations = restaurants.location.fillna('').str.lower().str.strip().tolist()
    nonveg = np.array([any(k in text for k in ['seafood','biryani','kebab','kabab','meat','steak','fish','chicken','mughlai']) for text in texts])
    costs = restaurants.cost_for_two.to_numpy()
    rating = restaurants.rating.fillna(restaurants.rating.median()).to_numpy() / 5
    popularity = restaurants.log_votes.to_numpy() / restaurants.log_votes.max()
    frames = []
    parse = lambda text: {c.strip().lower() for c in str(text).split(',') if c.strip()} if pd.notna(text) else set()
    for gid, members in groups.groupby('group_id'):
        sets = [parse(x) for x in members.preferred_cuisines]
        pairs = [len(a & b) / len(a | b) if a | b else 0 for i,a in enumerate(sets) for b in sets[i+1:]]
        values = {c: [] for c in ['cuisine_match','budget_match','dietary_compatibility','restaurant_type_match','location_match','booking_match','online_order_match']}
        for member in members.to_dict('records'):
            pref = parse(member['preferred_cuisines'])
            values['cuisine_match'].append([len(pref & c) / len(pref) if pref else 0 for c in cuisines])
            ratio = costs / member['budget']
            values['budget_match'].append(np.where(np.isnan(costs), .5, np.select([ratio<=1,ratio<=1.25,ratio<=1.5,ratio<=2],[1,.75,.5,.25],0)))
            diet = member['dietary_preference'].lower().strip()
            ds = np.ones(len(restaurants)) if diet=='no preference' else np.where(nonveg, .5,1) if diet=='vegetarian' else np.where(nonveg,.25,.75) if diet=='vegan' else np.where(nonveg,1,.75)
            values['dietary_compatibility'].append(ds)
            pt = str(member['preferred_restaurant_type']).lower().strip()
            values['restaurant_type_match'].append([1 if pt in [s.strip() for s in t.split(',')] else .5 if t and any(pt in s or s in pt for s in t.split(',')) else 0 for t in types])
            pl = str(member['preferred_location']).lower().strip()
            values['location_match'].append([float(pl==l) for l in locations])
            values['booking_match'].append(restaurants.book_table.to_numpy() if member['requires_table_booking']==1 else np.ones(len(restaurants)))
            values['online_order_match'].append(restaurants.online_order.to_numpy() if member['prefers_online_order']==1 else np.ones(len(restaurants)))
        f = pd.DataFrame({k: np.mean(v,axis=0) for k,v in values.items()})
        f['group_id'], f['group_size'] = gid,len(members)
        f['group_agreement'] = np.mean(pairs) if pairs else 0
        f['rating_score'], f['popularity_score'] = rating,popularity
        f['cost_for_two'] = restaurants.cost_for_two.fillna(restaurants.cost_for_two.median())
        for c in ['votes','online_order','book_table']: f[c]=restaurants[c]
        frames.append(f)
    frame = pd.concat(frames,ignore_index=True)
    frame['target'] = np.clip(sum(frame[k]*v for k,v in TARGET_WEIGHTS.items()) + np.random.RandomState(42).normal(0,.03,len(frame)),0,1)
    gids = frame.group_id.unique()
    np.random.RandomState(42).shuffle(gids)
    test = frame[frame.group_id.isin(gids[int(.8*len(gids)):])]
    pred = test[FEATURES].to_numpy() @ model.coef_ + model.intercept_
    result = {'scope': 'forensic replay of original corrected target and legacy builder, not raw pipeline validation',
              'metrics': metrics(test.target,pred), 'test_rows':len(test), 'test_groups':sorted(test.group_id.unique().tolist()),
              'target_weights':TARGET_WEIGHTS, 'snapshot_sha256':sha256(backup),
              'model_original_sklearn_version':'1.7.2 (from deserialization warning)',
              'method':'saved linear coefficients and intercept evaluated with numpy; no training',
              'warnings':[str(w.message) for w in captured],
              'historical_csv_r2':0.7641078524673492,
              'unresolved':'Neither original split nor original matrix is supplied; discrepancy attribution requires those artifacts or an immutable original run manifest.'}
    write_json(ROOT / 'SmartDine/outputs/historical_replay.json',result)
    print(result['metrics'])
    return result


if __name__=='__main__':
    replay()
