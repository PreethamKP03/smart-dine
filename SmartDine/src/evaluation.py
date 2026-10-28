"""Re-evaluate the exact hashed split and generate benchmark diagnostics."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .paths import DATA_DIR, MODEL_DIR, OUTPUT_DIR, FIGURE_DIR
from .schema import FEATURES, write_json
from .metadata import load_metadata
from .train_model import load_splits, metrics


def evaluate_system(data_dir=DATA_DIR, model_dir=MODEL_DIR, output_dir=OUTPUT_DIR):
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    train, test, manifest = load_splits(data_dir)
    meta = load_metadata(model_dir / 'model_metadata.json', model_dir / 'smartdine_best_model.joblib')
    if meta['status'] != 'available' or meta['split']['test_sha256'] != manifest['test_sha256']:
        raise ValueError('Verified metadata for this model/test split is required; train first.')
    model = joblib.load(model_dir / 'smartdine_best_model.joblib')
    pred = model.predict(test[FEATURES])
    actual = test.suitability_score.to_numpy()
    result = metrics(actual, pred)
    pd.DataFrame([{'Metric': k, 'Value': v} for k, v in result.items()]).to_csv(output_dir / 'model_evaluation_metrics.csv', index=False)
    figures = output_dir / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(actual[:5000], pred[:5000], s=8, alpha=.2)
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set(xlabel='Constructed benchmark target', ylabel='Predicted index', title='Held-out groups: first 5,000 pairs')
    fig.tight_layout(); fig.savefig(figures / 'actual_vs_predicted.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(actual - pred, bins=50)
    ax.set(xlabel='Constructed target minus prediction', ylabel='Pair count', title='Held-out benchmark residuals')
    fig.tight_layout(); fig.savefig(figures / 'residual_distribution.png', dpi=150); plt.close(fig)
    if hasattr(model, 'coef_'):
        coefficients = pd.DataFrame({'Feature': FEATURES, 'Coefficient': model.coef_})
        coefficients.to_csv(output_dir / 'feature_coefficients.csv', index=False)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(coefficients.Feature, coefficients.Coefficient)
        ax.set(xlabel='Coefficient (features have different units)', title='Linear model coefficients; not causal effects')
        fig.tight_layout(); fig.savefig(figures / 'feature_coefficients.png', dpi=150); plt.close(fig)
    # Metadata must agree with independent re-evaluation.
    if any(not np.isclose(result[k], meta['metrics'][k], rtol=1e-9, atol=1e-12) for k in result):
        raise ValueError('Recomputed metrics disagree with model metadata.')
    print(f'Recomputed test metrics: {result}')
    return result


if __name__ == '__main__':
    evaluate_system()
