from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / 'SmartDine'
DATA_DIR = BASE_DIR / 'data' / 'processed'
RAW_PATH = BASE_DIR / 'data' / 'raw' / 'zomato.csv'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs'
FIGURE_DIR = OUTPUT_DIR / 'figures'
MODEL_PATH = MODEL_DIR / 'smartdine_best_model.joblib'
METADATA_PATH = MODEL_DIR / 'model_metadata.json'
FEATURE_PATH = MODEL_DIR / 'feature_columns.txt'
