"""Fail closed for absent, corrupt, stale, or unverified metric metadata."""
import json
import math
from pathlib import Path
from .schema import FEATURES, sha256
from .paths import METADATA_PATH, MODEL_PATH


def load_metadata(path=METADATA_PATH, model_path=MODEL_PATH):
    unavailable = {'status': 'unavailable', 'metrics': None}
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(data, dict) or data.get('status') != 'available':
            return unavailable
        if data.get('feature_columns') != FEATURES or data.get('model_sha256') != sha256(model_path):
            return unavailable
        metrics = data['metrics']
        if not all(isinstance(metrics[k], (int, float)) and math.isfinite(metrics[k]) for k in ['MAE', 'RMSE', 'R2']):
            return unavailable
        return data
    except (OSError, ValueError, TypeError, KeyError):
        return unavailable
