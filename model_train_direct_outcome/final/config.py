from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SAMPLE_DIR = ROOT / "sample"
RESULTS_ROOT = ROOT / "results" / "final_pipeline"
POSITIONS = ("AV", "MV", "PV", "TV")

TRAIN_CSV = ROOT / "data" / "circor" / "training_data.csv"
TRAIN_FEATURE_CSV = ROOT / "data" / "circor" / "hms_features_final_single_v1.csv"
HMS_CHECKPOINT = ROOT / "checkpoints" / "final_hms_single_v1.pth"

OUTCOME_FEATURE_SET = "base_timing_grade_shape_position_persistent_no_embedding"
OUTCOME_MODEL_PRESET = "single_final_v1"
OUTCOME_THRESHOLD = 0.54

TRIAGE_LOW_THRESHOLD = 0.18
TRIAGE_WATCH_THRESHOLD = 0.40
TRIAGE_HIGH_THRESHOLD = 0.54

AGE_CHOICES = (
    "Neonate",
    "Infant",
    "Child",
    "Adolescent",
    "Adult",
)
SEX_CHOICES = (
    "Female",
    "Male",
)

QUALITY_DIRECT_LEVELS = {"good", "borderline"}
QUALITY_CLEAN_LEVELS = {"needs_light_cleaning"}
QUALITY_REVIEW_LEVELS = {"needs_cleaning", "poor"}

CLEANING_REFERENCE_DIR = ROOT / "data" / "circor" / "training_data"
CLEANING_REFERENCE_LIMIT = 240
CLEANING_HIGHPASS_HZ = 20.0
CLEANING_LOWPASS_HZ = 900.0
CLEANING_FILTER_ORDER = 2
CLEANING_MAX_BAND_GAIN = 1.6
CLEANING_TARGET_MIX = 0.18
