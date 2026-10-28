import numpy as np
import pytest
from SmartDine.src.group_preferences import MemberPreference as M, GroupPreferences as G


@pytest.mark.parametrize('budget', [0, -1, np.nan, np.inf])
def test_invalid_budget(budget):
    with pytest.raises(ValueError, match='Budget'):
        M(budget=budget)


@pytest.mark.parametrize('kwargs', [{'min_rating':6}, {'cuisine_importance':-1},
                                  {'budget_importance':np.nan}, {'dietary_preference':'allergy-safe'}])
def test_invalid_member_fields(kwargs):
    with pytest.raises(ValueError):
        M(**kwargs)


def test_empty_group():
    with pytest.raises(ValueError): G([])


def test_consensus_and_weighted_budget():
    group = G([M(preferred_cuisines=['Italian', 'Chinese'], budget=300, budget_importance=1),
               M(preferred_cuisines=['Italian'], budget=900, budget_importance=3)])
    profile = group.compute_group_profile()
    assert profile['group_agreement'] == .5
    assert profile['weighted_budget'] == 750
    assert G([M(), M()]).compute_group_profile()['group_agreement'] == 0


def test_duplicate_cuisine_is_not_double_counted():
    assert M(preferred_cuisines=['Italian','italian']).preferred_cuisines == ['italian']


def test_zero_weights_use_equal_weight_fallback():
    assert G([M(budget=100, budget_importance=0), M(budget=300, budget_importance=0)]).compute_group_profile()['weighted_budget'] == 200


def test_contrasting_groups(engine):
    italian = engine.recommend(G([M(preferred_cuisines=['Italian'], min_rating=0)]))
    chinese = engine.recommend(G([M(preferred_cuisines=['Chinese'], min_rating=0)]))
    assert italian['recommendations'][0]['name'] == 'Bistro'
    assert chinese['recommendations'][0]['name'] == 'Budget'


def test_strict_filters_no_silent_relaxation(engine):
    result = engine.recommend(G([M()]), filter_location='missing place')
    assert result['recommendations'] == []
    assert result['filters_relaxed'] is False and result['message']


@pytest.mark.parametrize('filters', [{'max_budget':100}, {'min_rating':5},
                                   {'filter_cuisine':'impossible'}, {'filter_location':'Nor'}])
def test_no_results(engine, filters):
    assert engine.recommend(G([M()]), **filters)['recommendations'] == []


def test_booking_and_budget_filters(engine):
    result = engine.recommend(G([M(min_rating=0)]), require_booking=True, max_budget=1000)
    assert [r['name'] for r in result['recommendations']] == ['Bistro']
    assert engine.recommend(G([M(min_rating=0)]), require_booking=True, max_budget=500)['recommendations'] == []


def test_member_booking_requirement(engine):
    assert all(r['book_table'] is True for r in engine.recommend(G([M(requires_booking=True)]))['recommendations'])


def test_unknown_fields_not_confirmed(engine):
    result = engine.recommend(G([M(min_rating=0)]), top_n=3)
    unknown = next(r for r in result['recommendations'] if r['name']=='Unknown')
    assert unknown['cost_for_two'] is None and unknown['book_table'] is None
    assert unknown['rating'] is None
    assert 'Cost is unknown' in unknown['explanation']


def test_dietary_explanation_unverified(engine):
    result = engine.recommend(G([M(dietary_preference='vegan')]))
    for row in result['recommendations']:
        assert row['dietary_status'] == 'unverified'
        assert 'unverified' in row['explanation']
        assert 'Accommodates dietary' not in row['explanation']


def test_missing_preferences_do_not_invent_location_reason(engine):
    for row in engine.recommend(G([M()]))['recommendations']:
        assert 'Neighborhood name match' not in row['explanation']
        assert 'Cuisine preference coverage' not in row['explanation']


def test_budget_explanation_checks_all_members(engine):
    result = engine.recommend(G([M(budget=800), M(budget=1000)]), filter_location='North')
    assert 'exceeds at least one' in result['recommendations'][0]['explanation']
    assert 'within every member' not in result['recommendations'][0]['explanation']


def test_repeat_and_tie_order(engine):
    engine.model.coef_[:] = 0
    result = engine.recommend(G([M(min_rating=0)]), top_n=3)
    assert [r['restaurant_id'] for r in result['recommendations']] == ['a','b','c']
    assert result == engine.recommend(G([M(min_rating=0)]), top_n=3)


@pytest.mark.parametrize('kwargs', [{'top_n':0},{'top_n':-1},{'max_budget':0},{'max_budget':np.nan},{'min_rating':6}])
def test_invalid_recommendation_inputs(engine, kwargs):
    with pytest.raises(ValueError): engine.recommend(G([M()]), **kwargs)
