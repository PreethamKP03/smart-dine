"""Import-safe restaurant feature generation; unknown values remain unknown."""
from pathlib import Path
import numpy as np
import pandas as pd
from .paths import DATA_DIR


def engineer_features(frame):
    required = {'restaurant_id', 'name', 'rating', 'cost_for_two', 'votes', 'cuisines', 'book_table', 'online_order'}
    if not required <= set(frame):
        raise ValueError(f'Missing cleaned columns: {sorted(required - set(frame))}')
    df = frame.copy()
    df['log_votes'] = np.log1p(pd.to_numeric(df.votes, errors='coerce').clip(lower=0))
    df['cuisine_count'] = df.cuisines.fillna('').map(lambda x: len({c.strip() for c in str(x).split(',') if c.strip()}))
    df['has_rating'], df['has_cost'] = df.rating.notna().astype(int), df.cost_for_two.notna().astype(int)
    df['has_booking'], df['has_online_order'] = df.book_table, df.online_order
    cols = ['restaurant_id', 'name', 'location', 'cuisines', 'rest_type', 'listed_in_type',
            'rating', 'votes', 'log_votes', 'cost_for_two', 'online_order', 'book_table',
            'cuisine_count', 'has_rating', 'has_cost', 'has_booking', 'has_online_order']
    for key in cols:
        if key not in df: df[key] = None
    return df[cols]


def generate_features(input_path=None, output_path=None):
    result = engineer_features(pd.read_csv(input_path or DATA_DIR / 'restaurants_cleaned.csv'))
    dest = Path(output_path or DATA_DIR / 'restaurant_features.csv')
    dest.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(dest, index=False)
    print(f'Generated features for {len(result)} restaurants.')
    return result


if __name__ == '__main__':
    generate_features()
