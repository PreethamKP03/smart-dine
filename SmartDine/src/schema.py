"""Single versioned feature and constructed-target contract."""
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = '2.0'
FEATURES = ['group_size', 'cuisine_match', 'budget_match', 'dietary_compatibility',
            'restaurant_type_match', 'location_match', 'booking_match',
            'online_order_match', 'group_agreement', 'rating_score',
            'popularity_score', 'cost_for_two', 'votes', 'online_order', 'book_table']
# Preserve the corrected target weights in the supplied prepare_ml_dataset.py.
TARGET_WEIGHTS = {'cuisine_match': .20, 'budget_match': .20,
                  'dietary_compatibility': .15, 'restaurant_type_match': .10,
                  'location_match': .10, 'group_agreement': .10,
                  'rating_score': .05, 'booking_match': .03,
                  'online_order_match': .03, 'popularity_score': .04}
TARGET = {'kind': 'constructed benchmark; not human preference or satisfaction',
          'weights': TARGET_WEIGHTS, 'noise': 'numpy RandomState(42), normal(0, 0.03)',
          'clip': [0, 1], 'version': VERSION}


def suitability_target(frame, seed=42):
    signal = sum(frame[c].to_numpy(dtype=float) * w for c, w in TARGET_WEIGHTS.items())
    return np.clip(signal + np.random.RandomState(seed).normal(0, .03, len(frame)), 0, 1)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
