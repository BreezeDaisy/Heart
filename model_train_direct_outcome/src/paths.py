from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT / "data" / "circor"
CHECKPOINT_ROOT = PACKAGE_ROOT / "checkpoints"
RESULTS_ROOT = PACKAGE_ROOT / "results"

TRAIN_CSV = DATA_ROOT / "training_data.csv"
TRAIN_WAV_DIR = DATA_ROOT / "training_data"
POSITION_FALLBACK_CSV = DATA_ROOT / "position_murmur_features.csv"

FINAL_HMS_CHECKPOINT = CHECKPOINT_ROOT / "final_hms_single_v1.pth"
FINAL_HMS_METADATA = CHECKPOINT_ROOT / "final_hms_single_v1.json"
FINAL_FEATURE_CSV = DATA_ROOT / "hms_features_final_single_v1.csv"
