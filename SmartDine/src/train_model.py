"""Validation-selected regression models, untouched group test holdout, honest metadata."""
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from .paths import DATA_DIR, MODEL_DIR, OUTPUT_DIR
from .schema import FEATURES, TARGET, VERSION, sha256, write_json


def metrics(actual, predicted):
    return {'MAE': float(mean_absolute_error(actual, predicted)),
            'RMSE': float(np.sqrt(mean_squared_error(actual, predicted))),
            'R2': float(r2_score(actual, predicted))}


def load_splits(data_dir=DATA_DIR):
    root = Path(data_dir)
    manifest = json.loads((root / 'split_manifest.json').read_text(encoding='utf-8'))
    if manifest['feature_version'] != VERSION or manifest['target'] != TARGET:
        raise ValueError('Stale target/feature version; regenerate matrix and split.')
    if root.resolve() == DATA_DIR.resolve():
        for key, filename in [('restaurant_sha256', 'restaurant_features.csv'), ('groups_sha256', 'group_members.csv')]:
            if manifest['matrix_inputs'][key] != sha256(DATA_DIR / filename):
                raise ValueError('Split source inputs changed; regenerate matrix and split.')
    for name, key in [('train_ml.csv', 'train_sha256'), ('test_ml.csv', 'test_sha256')]:
        if sha256(root / name) != manifest[key]:
            raise ValueError('Split content changed; regenerate with pipeline.py split.')
    train, test = pd.read_csv(root / 'train_ml.csv'), pd.read_csv(root / 'test_ml.csv')
    if train.empty or len(test) < 2 or set(train.group_id) & set(test.group_id):
        raise ValueError('Empty splits or train/test group leakage.')
    for frame, label in [(train, 'train'), (test, 'test')]:
        if frame.group_id.isna().any() or frame[['group_id', 'restaurant_id']].duplicated().any():
            raise ValueError('Missing groups or duplicate group/restaurant pairs.')
        if sorted(frame.group_id.unique().tolist()) != manifest[label + '_groups']:
            raise ValueError('Split groups disagree with manifest.')
        if not np.isfinite(frame[FEATURES + ['suitability_score']].to_numpy()).all():
            raise ValueError('Non-finite features or labels.')
    return train, test, manifest


def train_and_compare_models(data_dir=DATA_DIR, model_dir=MODEL_DIR, output_dir=OUTPUT_DIR):
    train, test, manifest = load_splits(data_dir)
    validation_split = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
    fit_idx, val_idx = next(validation_split.split(train, groups=train.group_id))
    candidates = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=2),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=60, max_depth=5, learning_rate=.1, random_state=42)}
    rows = []
    for name, model in candidates.items():
        started = time.perf_counter()
        model.fit(train.iloc[fit_idx][FEATURES], train.iloc[fit_idx].suitability_score)
        result = metrics(train.iloc[val_idx].suitability_score, model.predict(train.iloc[val_idx][FEATURES]))
        rows.append({'Model': name, **result, 'Training_Time_Seconds': time.perf_counter() - started})
        print(f'Validation {name}: {result}', flush=True)
    comparison = pd.DataFrame(rows).sort_values(['R2', 'MAE'], ascending=[False, True])
    champion = comparison.iloc[0].Model
    model = candidates[champion]
    model.fit(train[FEATURES], train.suitability_score)
    predicted = model.predict(test[FEATURES])
    result = metrics(test.suitability_score, predicted)
    baseline = DummyRegressor(strategy='mean').fit(train[FEATURES], train.suitability_score)
    baseline_result = metrics(test.suitability_score, baseline.predict(test[FEATURES]))
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'smartdine_best_model.joblib'
    joblib.dump(model, model_path)
    (model_dir / 'feature_columns.txt').write_text('\n'.join(FEATURES) + '\n', encoding='utf-8')
    comparison.to_csv(output_dir / 'model_comparison.csv', index=False)
    metadata = {'status': 'available', 'model_name': champion, 'estimator': type(model).__name__,
                'feature_columns': FEATURES, 'feature_version': VERSION, 'target_definition': TARGET,
                'metrics': result, 'baseline': {'definition': 'constant training-target mean', 'metrics': baseline_result},
                'selection': 'highest validation R2, then lowest MAE; test used only after selection',
                'validation_groups': sorted(train.iloc[val_idx].group_id.unique().tolist()),
                'fit_groups': sorted(train.iloc[fit_idx].group_id.unique().tolist()),
                'split': manifest, 'model_sha256': sha256(model_path),
                'python_version': __import__('platform').python_version(), 'sklearn_version': sklearn.__version__,
                'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                'raw_pipeline_verified': False,
                'limitations': 'Synthetic target and groups. No human evaluation. Processed source provenance not independently verified.'}
    write_json(model_dir / 'model_metadata.json', metadata)
    print(f'Test {champion}: {result}; training-mean baseline: {baseline_result}', flush=True)
    return champion, comparison


if __name__ == '__main__':
    train_and_compare_models()
