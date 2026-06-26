import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
DEFAULT_UPLOAD_DIR = Path("/tmp/lungsphere_uploads") if os.environ.get("VERCEL") else BASE_DIR / "uploads"
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", DEFAULT_UPLOAD_DIR))

FEATURES_CSV = DATA_DIR / "fraiwan_dwt_segments_best.csv"
SVM_MODEL_PATH = MODEL_DIR / "svm_dwt_mfcc_lungsphere.joblib"
RF_MODEL_PATH = MODEL_DIR / "rf_dwt_mfcc_lungsphere.joblib"
BEST_MODEL_PATH = MODEL_DIR / "best_lungsphere_model.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "best_lungsphere_model_metadata.json"
EVALUATION_REPORT_PATH = REPORT_DIR / "model_evaluation.json"

TARGET_SR = 4000
SEGMENT_DURATION = 4.0
WAVELET = "db4"
DWT_LEVEL = 4
N_MFCC = 20

# Default class-probability threshold for segment-level predictions.
DECISION_THRESHOLD = 0.5
RANDOM_STATE = 42

# Feature order in fraiwan_dwt_segments_best.csv:
# feat_0..feat_29 = DWT statistics, feat_30..feat_69 = MFCC mean/std features.
MFCC_FEATURE_START = 30
MFCC_FEATURE_END = 70
FEATURE_COUNT = 70
