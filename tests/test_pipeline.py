import json
import pandas as pd
import pytest
from SmartDine.src.build_group_restaurant_features import build_matrix
from SmartDine.src.prepare_ml_dataset import prepare_dataset
from SmartDine.src.train_model import load_splits, train_and_compare_models
from SmartDine.src.metadata import load_metadata
from SmartDine.src.schema import sha256


@pytest.fixture
def tiny_pipeline(tmp_path, restaurants):
    rp, gp = tmp_path/'restaurants.csv', tmp_path/'groups.csv'
    restaurants.to_csv(rp,index=False)
    # Test-only synthetic preferences; no raw restaurant replacement.
    pd.DataFrame([{'group_id':f'G{i}', 'budget':200+100*i,
                   'preferred_cuisines':'Italian' if i%2 else 'Chinese',
                   'dietary_preference':'no preference'} for i in range(10)]).to_csv(gp,index=False)
    matrix = build_matrix(rp,gp,tmp_path/'matrix.csv')
    prepare_dataset(matrix,tmp_path)
    return tmp_path


def test_matrix_repeat_and_single_target_generation(tiny_pipeline):
    root=tiny_pipeline
    first=(root/'matrix.csv').read_bytes()
    build_matrix(root/'restaurants.csv',root/'groups.csv',root/'matrix.csv')
    assert (root/'matrix.csv').read_bytes()==first
    train,test,manifest=load_splits(root)
    assert not set(train.group_id)&set(test.group_id)
    assert len(train)+len(test)==30


def test_tampered_matrix_rejected(tiny_pipeline):
    root=tiny_pipeline
    with (root/'matrix.csv').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='Stale matrix'):
        prepare_dataset(root/'matrix.csv',root)


def test_tampered_split_rejected(tiny_pipeline):
    root=tiny_pipeline
    with (root/'test_ml.csv').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='Split content changed'):
        load_splits(root)


def test_group_overlap_rejected_even_with_updated_hash(tiny_pipeline):
    root=tiny_pipeline
    train=pd.read_csv(root/'train_ml.csv')
    test=pd.read_csv(root/'test_ml.csv')
    test['group_id']=train.group_id.iloc[0]
    test.to_csv(root/'test_ml.csv',index=False)
    path=root/'split_manifest.json'
    manifest=json.loads(path.read_text())
    manifest['test_sha256']=sha256(root/'test_ml.csv')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='group leakage'):
        load_splits(root)


def test_train_metadata_validation_and_test_separation(tiny_pipeline):
    root=tiny_pipeline
    model_dir=root/'models'
    train_and_compare_models(root,model_dir,root/'outputs')
    meta=load_metadata(model_dir/'model_metadata.json',model_dir/'smartdine_best_model.joblib')
    assert meta['status']=='available'
    assert not set(meta['validation_groups']) & set(meta['fit_groups'])
    assert not set(meta['split']['test_groups']) & set(meta['split']['train_groups'])
    assert set(meta['metrics'])=={'MAE','RMSE','R2'}
