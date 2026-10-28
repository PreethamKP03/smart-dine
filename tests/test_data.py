import numpy as np
import pandas as pd
import pytest
from SmartDine.src.preprocessing import clean_frame, validate_raw, clean_dataset
from SmartDine.src.feature_engineering import engineer_features, generate_features
from SmartDine.src.prepare_ml_dataset import group_split, prepare_dataset
from SmartDine.src.features import RestaurantFeatureBuilder
from SmartDine.src.schema import FEATURES, suitability_target
from SmartDine.src.group_preferences import GroupPreferences, MemberPreference


def raw_rows():
    row = {'name': ' Cafe ', 'address': ' 1 Street ', 'location': 'North',
           'cuisines': 'Italian', 'rate': '4.0/5', 'votes': '10',
           'approx_cost(for two people)': '1,000', 'online_order': 'Yes', 'book_table': 'No'}
    return pd.DataFrame([row, {**row, 'rate': 'NEW', 'votes': '20'},
                         {**row, 'address': '2 Street'}, {**row, 'name': None}])


def test_consolidation_preserves_branches_and_counts():
    result, report = clean_frame(raw_rows())
    assert len(result) == 2
    first = result[result.address.eq('1 Street')].iloc[0]
    assert first.source_record_count == 2
    assert first.votes == 20 and first.rating == 4 and first.cost_for_two == 1000
    assert report['removed_missing_identity'] == 1
    assert report['consolidated_rows'] == 1
    assert report['removed_zero_based_row_indices'] == [3]


def test_unknown_and_conflicting_flags():
    raw = raw_rows().iloc[:2].copy()
    raw.loc[1, 'online_order'] = 'No'
    raw['book_table'] = 'maybe'
    result, report = clean_frame(raw)
    assert result.book_table.isna().all() and result.online_order.isna().all()
    assert report['conflicting_boolean_entities'] == 1


@pytest.mark.parametrize('rating,cost,votes', [('9/5', '-1', '-2'), ('NEW', 'unknown', 'none')])
def test_invalid_numbers_stay_unknown(rating, cost, votes):
    raw = raw_rows().iloc[:1].copy()
    raw['rate'], raw['approx_cost(for two people)'], raw['votes'] = rating, cost, votes
    result, _ = clean_frame(raw)
    assert result[['rating', 'cost_for_two', 'votes']].isna().all().all()


def test_missing_raw(tmp_path):
    with pytest.raises(FileNotFoundError, match='Original Zomato'):
        validate_raw(tmp_path / 'absent.csv')


def test_malformed_raw(tmp_path):
    path = tmp_path / 'bad.csv'
    path.write_text('name\na\n')
    with pytest.raises(ValueError, match='Malformed'):
        validate_raw(path)


def test_raw_unchanged_and_provenance(tmp_path):
    raw = tmp_path / 'raw.csv'
    raw_rows().to_csv(raw, index=False)
    before = raw.read_bytes()
    dest = tmp_path / 'clean.csv'
    clean_dataset(raw, dest)
    assert raw.read_bytes() == before and dest.with_suffix('.provenance.json').is_file()
    with pytest.raises(ValueError, match='overwrite'):
        clean_dataset(raw, raw)


def test_headers_and_entity_case():
    raw = raw_rows().iloc[:2].copy()
    raw.loc[1, 'name'] = 'CAFE'
    raw.columns = [' ' + c.upper() + ' ' for c in raw.columns]
    assert len(clean_frame(raw)[0]) == 1


def test_restaurant_features_unknowns():
    clean, _ = clean_frame(raw_rows())
    clean.loc[0, 'book_table'] = np.nan
    features = engineer_features(clean)
    assert np.isnan(features.loc[0, 'has_booking'])
    assert features.cuisine_count.eq(1).all()
    assert np.allclose(features.log_votes, np.log1p(clean.votes))


def test_missing_feature_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_features(tmp_path / 'missing.csv', tmp_path / 'out.csv')


def test_group_split_separation():
    frame = pd.DataFrame({'group_id': np.repeat(['A','B','C','D','E'], 2), 'restaurant_id': ['r1','r2']*5})
    train, test = group_split(frame)
    assert not set(train.group_id) & set(test.group_id)
    assert len(train) + len(test) == len(frame)
    assert group_split(frame)[0].equals(train)


def test_split_rejects_duplicate_pairs():
    with pytest.raises(ValueError, match='duplicate'):
        group_split(pd.DataFrame({'group_id': ['A']*3, 'restaurant_id': ['r']*3}))


def test_split_rejects_too_few_groups():
    with pytest.raises(ValueError, match='three'):
        group_split(pd.DataFrame({'group_id': ['A','B'], 'restaurant_id': ['r','r']}))


def test_missing_manifest_prevents_stale_split(tmp_path):
    path = tmp_path / 'matrix.csv'
    path.write_text('unrelated\n1\n')
    with pytest.raises(FileNotFoundError):
        prepare_dataset(path, tmp_path)


def test_shared_features_finite_exact_schema(restaurants):
    builder = RestaurantFeatureBuilder(restaurants)
    group = GroupPreferences([MemberPreference(budget=500)])
    frame = builder.build(group)
    assert frame.columns.tolist() == FEATURES
    assert np.isfinite(frame.to_numpy()).all()
    assert frame.iloc[2].budget_match == .5
    assert np.isnan(builder.df.iloc[2].cost_for_two)


def test_missing_identity_rejected(restaurants):
    with pytest.raises(ValueError):
        RestaurantFeatureBuilder(restaurants.drop(columns='restaurant_id'))


def test_duplicate_identity_rejected(restaurants):
    restaurants.loc[1, 'restaurant_id'] = 'b'
    with pytest.raises(ValueError, match='unique'):
        RestaurantFeatureBuilder(restaurants)


def test_target_reproducible(restaurants):
    x = RestaurantFeatureBuilder(restaurants).build(GroupPreferences([MemberPreference()]))
    assert np.array_equal(suitability_target(x), suitability_target(x))
    assert ((suitability_target(x) >= 0) & (suitability_target(x) <= 1)).all()
