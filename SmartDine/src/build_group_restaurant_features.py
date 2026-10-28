"""Build the real supplied catalogue x synthetic groups with the serving feature code."""
from pathlib import Path
import pandas as pd
from .paths import DATA_DIR
from .features import RestaurantFeatureBuilder, members_from_frame
from .group_preferences import GroupPreferences
from .schema import VERSION, sha256, write_json


def build_matrix(restaurant_path=None, group_path=None, output_path=None):
    rp = Path(restaurant_path or DATA_DIR / 'restaurant_features.csv')
    gp = Path(group_path or DATA_DIR / 'group_members.csv')
    dest = Path(output_path or DATA_DIR / 'group_restaurant_features.csv')
    builder, groups = RestaurantFeatureBuilder(pd.read_csv(rp)), pd.read_csv(gp)
    if 'group_id' not in groups or groups.empty or groups.group_id.isna().any():
        raise ValueError('Nonempty group_id column required.')
    dest.parent.mkdir(parents=True, exist_ok=True)
    first, count = True, 0
    temporary = dest.with_suffix('.csv.tmp')
    for gid, frame in groups.groupby('group_id', sort=True):
        features = builder.build(GroupPreferences(members_from_frame(frame)))
        features.insert(0, 'restaurant_id', builder.df.restaurant_id.values)
        features.insert(0, 'group_id', gid)
        features.to_csv(temporary, mode='w' if first else 'a', header=first, index=False)
        first, count = False, count + len(features)
    temporary.replace(dest)
    write_json(dest.with_suffix('.manifest.json'), {'feature_version': VERSION,
               'restaurant_sha256': sha256(rp), 'groups_sha256': sha256(gp),
               'matrix_sha256': sha256(dest), 'rows': count,
               'groups': int(groups.group_id.nunique()), 'restaurants': len(builder.df),
               'target': 'generated once during split, not here'})
    print(f'Built {count} pairs from {groups.group_id.nunique()} groups and {len(builder.df)} restaurants.')
    return dest


if __name__ == '__main__':
    build_matrix()
