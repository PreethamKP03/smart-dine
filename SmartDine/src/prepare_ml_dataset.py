"""Generate one target and disjoint group train/test partitions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .paths import DATA_DIR
from .schema import FEATURES, TARGET, VERSION, suitability_target, sha256, write_json


def group_split(df, seed=42):
    if df.group_id.isna().any() or df[['group_id', 'restaurant_id']].duplicated().any():
        raise ValueError('Missing group IDs or duplicate group/restaurant pairs.')
    ids = np.array(sorted(df.group_id.unique()))
    if len(ids) < 3:
        raise ValueError('At least three groups required for holdout and validation.')
    np.random.RandomState(seed).shuffle(ids)
    split = max(1, min(len(ids)-1, int(len(ids) * .8)))
    return df[df.group_id.isin(ids[:split])].copy(), df[df.group_id.isin(ids[split:])].copy()


def prepare_dataset(input_path=None, output_dir=None):
    source = Path(input_path or DATA_DIR / 'group_restaurant_features.csv')
    dest = Path(output_dir or DATA_DIR)
    manifest = json.loads(source.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    if manifest['feature_version'] != VERSION or manifest['matrix_sha256'] != sha256(source):
        raise ValueError('Stale matrix; regenerate with pipeline.py matrix.')
    if source.parent.resolve() == DATA_DIR.resolve():
        for key, filename in [('restaurant_sha256', 'restaurant_features.csv'), ('groups_sha256', 'group_members.csv')]:
            if manifest[key] != sha256(DATA_DIR / filename):
                raise ValueError('Matrix source inputs changed; regenerate matrix.')
    frame = pd.read_csv(source)
    if not np.isfinite(frame[FEATURES].to_numpy()).all():
        raise ValueError('Non-finite ML features; regenerate matrix.')
    frame['suitability_score'] = suitability_target(frame)
    train, test = group_split(frame)
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in [('train_ml.csv', train), ('test_ml.csv', test)]:
        data.to_csv(dest / name, index=False)
    write_json(dest / 'split_manifest.json', {'feature_version': VERSION, 'target': TARGET,
               'matrix_sha256': sha256(source), 'matrix_inputs': manifest,
               'train_sha256': sha256(dest / 'train_ml.csv'), 'test_sha256': sha256(dest / 'test_ml.csv'),
               'train_groups': sorted(train.group_id.unique().tolist()),
               'test_groups': sorted(test.group_id.unique().tolist()),
               'train_rows': len(train), 'test_rows': len(test), 'seed': 42,
               'scope': 'unseen groups on the same known restaurant catalogue; restaurant overlap expected'})
    print(f'Train {len(train)} rows/{train.group_id.nunique()} groups; test {len(test)} rows/{test.group_id.nunique()} groups; overlap 0.')
    return train, test


if __name__ == '__main__':
    prepare_dataset()
