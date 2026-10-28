"""Raw input validation and conservative name + full-address entity consolidation."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .paths import RAW_PATH, DATA_DIR
from .schema import sha256, write_json

REQUIRED = {'name', 'address', 'location', 'cuisines', 'rate', 'votes',
            'approx_cost(for two people)', 'online_order', 'book_table'}


def validate_raw(path=RAW_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Original Zomato CSV missing. Supply SmartDine/data/raw/zomato.csv; no substitute is generated.')
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    if df.columns.duplicated().any():
        raise ValueError('Duplicate columns after header normalization.')
    missing = REQUIRED - set(df)
    if missing or df.empty:
        raise ValueError(f'Malformed raw dataset; missing columns: {sorted(missing)}, rows: {len(df)}')
    return df


def combine_unique_values(series):
    values = sorted({v.strip() for x in series.dropna() for v in str(x).split(',') if v.strip()})
    return ', '.join(values) if values else np.nan


def clean_frame(raw):
    df = raw.copy()
    df.columns = df.columns.str.strip().str.lower()
    missing = REQUIRED - set(df)
    if missing:
        raise ValueError(f'Missing raw columns: {sorted(missing)}')
    for c in df.select_dtypes(include=['object', 'string']):
        df[c] = df[c].astype('string').str.strip().replace('', pd.NA)
    for c in ['rest_type', 'dish_liked', 'menu_item', 'listed_in(type)', 'listed_in(city)']:
        if c not in df:
            df[c] = pd.NA
    before = len(df)
    valid = df.name.notna() & df.address.notna()
    rejected = df.index[~valid].tolist()
    df = df.loc[valid].copy()
    df['rating'] = pd.to_numeric(df.rate.str.extract(r'^(\d+(?:\.\d+)?)\s*(?:/\s*5)?$')[0], errors='coerce')
    df.loc[~df.rating.between(0, 5), 'rating'] = np.nan
    df['cost_for_two'] = pd.to_numeric(df['approx_cost(for two people)'].str.replace(',', '', regex=False), errors='coerce')
    df.loc[df.cost_for_two <= 0, 'cost_for_two'] = np.nan
    df['votes'] = pd.to_numeric(df.votes, errors='coerce')
    df.loc[df.votes < 0, 'votes'] = np.nan
    for c in ['online_order', 'book_table']:
        df[c] = df[c].str.lower().map({'yes': 1., 'no': 0.})
    # Collapse whitespace/case only; full address retained to protect distinct branches.
    norm = lambda s: s.str.lower().str.replace(r'\s+', ' ', regex=True)
    df['restaurant_id'] = norm(df.name) + ' | ' + norm(df.address)
    grouped = df.groupby('restaurant_id', sort=True)
    conflicts = int(((grouped.online_order.nunique() > 1) | (grouped.book_table.nunique() > 1)).sum())
    def boolean_consensus(series):
        values = series.dropna().unique()
        return float(values[0]) if len(values) == 1 else np.nan
    aggregation = {c: 'first' for c in ['name', 'address', 'location']}
    aggregation.update({c: combine_unique_values for c in ['cuisines', 'rest_type', 'dish_liked', 'menu_item', 'listed_in(type)', 'listed_in(city)']})
    aggregation.update({'rating': 'median', 'cost_for_two': 'median', 'votes': 'max',
                        'online_order': boolean_consensus, 'book_table': boolean_consensus})
    result = grouped.agg(aggregation).rename(columns={'listed_in(type)': 'listed_in_type', 'listed_in(city)': 'listed_in_city'})
    result['source_record_count'] = grouped.size()
    result = result.reset_index().sort_values(['name', 'restaurant_id']).reset_index(drop=True)
    report = {'input_rows': before, 'removed_missing_identity': len(rejected),
              'removed_zero_based_row_indices': rejected, 'retained_source_rows': len(df),
              'entities': len(result), 'consolidated_rows': len(df) - len(result),
              'conflicting_boolean_entities': conflicts,
              'missing_after_aggregation': result.isna().sum().to_dict(),
              'imputation': 'none', 'entity_key': 'normalized name + full address',
              'aggregation': 'rating/cost median, votes max (avoid repeated-listing inflation), text union; conflicting flags unknown',
              'caveat': 'Co-located identically named businesses can still merge; address variants may remain separate. No verified physical-entity ground truth.'}
    return result, report


def clean_dataset(raw_path=None, output_path=None):
    path, output = Path(raw_path or RAW_PATH), Path(output_path or DATA_DIR / 'restaurants_cleaned.csv')
    result, report = clean_frame(validate_raw(path))
    if output.resolve() == path.resolve():
        raise ValueError('Output must not overwrite original raw data.')
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    report['raw_sha256'] = sha256(path)
    write_json(output.with_suffix('.provenance.json'), report)
    print(f'Cleaned {report["input_rows"]} rows into {len(result)} entities.')
    return result


if __name__ == '__main__':
    clean_dataset()
