import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from SmartDine.src.schema import FEATURES, VERSION, sha256
from SmartDine.src.metadata import load_metadata
from SmartDine.src.recommendation import SmartDineRecommender
from SmartDine.src.paths import PROJECT_ROOT, DATA_DIR


def test_missing_model(engine, tmp_path):
    with pytest.raises(FileNotFoundError, match='Model missing'):
        SmartDineRecommender(engine.data_path, tmp_path / 'missing.joblib')


def test_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match='features missing'):
        SmartDineRecommender(tmp_path / 'absent.csv')


def test_model_schema_rejected(engine, tmp_path):
    path = tmp_path / 'wrong.joblib'
    joblib.dump(LinearRegression().fit(pd.DataFrame({'wrong':[0,1]}), [0,1]), path)
    with pytest.raises(ValueError, match='schema mismatch'):
        SmartDineRecommender(engine.data_path, path)


@pytest.mark.parametrize('content', [None, '{bad', '[]', '{}', '{"status":"available","metrics":null}'])
def test_missing_corrupt_metadata(tmp_path, content):
    path = tmp_path / 'metadata.json'
    if content is not None: path.write_text(content)
    assert load_metadata(path)['status'] == 'unavailable'
    assert load_metadata(path)['metrics'] is None


def test_verified_and_stale_metadata(engine, tmp_path):
    path = tmp_path / 'metadata.json'
    content = {'status':'available', 'feature_columns':FEATURES, 'feature_version':VERSION,
               'model_sha256':sha256(engine.model_path), 'metrics':{'MAE':.1,'RMSE':.2,'R2':.3}}
    path.write_text(json.dumps(content))
    assert load_metadata(path, engine.model_path)['metrics']['R2'] == .3
    content['model_sha256'] = 'stale'
    path.write_text(json.dumps(content))
    assert load_metadata(path, engine.model_path)['metrics'] is None


def test_root_relative_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert DATA_DIR == PROJECT_ROOT / 'SmartDine/data/processed'
    assert DATA_DIR.is_absolute()


@pytest.fixture
def client(monkeypatch, engine):
    from fastapi.testclient import TestClient
    from SmartDine.src import serving
    monkeypatch.setattr(serving, 'SmartDineRecommender', lambda: engine)
    # Tests do not report the real model metrics for a fixture engine.
    monkeypatch.setattr(serving, 'load_metadata', lambda: {'status':'unavailable','metrics':None})
    with TestClient(serving.app) as client:
        yield client


def test_api_health_unavailable_metrics(client):
    result = client.get('/api/health')
    assert result.status_code == 200
    assert result.json()['model_r2_score'] is None
    assert result.json()['metadata_status'] == 'unavailable'
    assert result.json()['total_restaurants'] == 3


def test_api_recommend(client):
    result = client.post('/api/recommend', json={'members':[{'preferred_cuisines':['Italian']}], 'top_n':1})
    assert result.status_code == 200
    assert result.json()['recommendations'][0]['name'] == 'Bistro'


@pytest.mark.parametrize('payload', [{'members':[]}, {'members':[{'budget':0}]},
                                    {'members':[{}], 'top_n':0}, {'members':[{}], 'min_rating':7},
                                    {'members':[{'cuisine_importance':-1}]},
                                    {'members':[{}], 'max_budget_limit':0}])
def test_api_bad_input(client, payload):
    assert client.post('/api/recommend', json=payload).status_code == 422


def test_api_empty_and_metadata(client):
    result = client.post('/api/recommend', json={'members':[{}], 'filter_location':'absent'})
    assert result.status_code == 200 and not result.json()['recommendations']
    assert result.json()['filters_relaxed'] is False
    assert client.get('/api/metadata').json()['model_metadata']['metrics'] is None
    assert client.get('/api/presets').status_code == 200


def test_dashboard_script_and_styles_are_served(client):
    html = client.get('/')
    assert html.status_code == 200
    assert 'src="/static/app.js?v=2"' in html.text
    assert 'href="/static/index.css?v=2"' in html.text
    assert client.get('/static/app.js').status_code == 200
    assert client.get('/static/index.css').status_code == 200


def test_api_failed_initialization(monkeypatch):
    from fastapi.testclient import TestClient
    from SmartDine.src import serving
    def missing(): raise FileNotFoundError('private path must not be exposed')
    monkeypatch.setattr(serving, 'SmartDineRecommender', missing)
    with TestClient(serving.app) as client:
        assert client.get('/api/health').status_code == 503
        response = client.post('/api/recommend', json={'members':[{}]})
        assert response.status_code == 503
        assert 'private path' not in response.text
