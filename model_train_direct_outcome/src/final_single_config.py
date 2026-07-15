from paths import FINAL_FEATURE_CSV, FINAL_HMS_CHECKPOINT, FINAL_HMS_METADATA


FINAL_VERSION = "final_single_persistent_no_embedding_v1"

# Final deployable single-model pipeline selected from local validation.
FINAL_HMS_SEED = 0
FINAL_HMS_CHECKPOINT = str(FINAL_HMS_CHECKPOINT)
FINAL_HMS_METADATA = str(FINAL_HMS_METADATA)
FINAL_FEATURE_CSV = str(FINAL_FEATURE_CSV)

FINAL_FEATURE_SET = "base_timing_grade_shape_position_persistent_no_embedding"
FINAL_MODEL_PRESET = "single_final_v1"
FINAL_THRESHOLD = 0.54
FINAL_OUTPUT_PREFIX = "outcome_hms_final_single_persistent_no_embedding_v1"

# Training defaults for continuing the single-model line.
FINAL_TRAIN_ARGS = [
    "--seed",
    str(FINAL_HMS_SEED),
    "--device",
    "cuda",
    "--epochs",
    "40",
    "--patience",
    "10",
    "--batch-size",
    "32",
    "--num-workers",
    "0",
    "--checkpoint-path",
    FINAL_HMS_CHECKPOINT,
]

# Proven source checkpoint that was promoted to the final single-model version.
FINAL_SOURCE_HMS_CHECKPOINT = "checkpoints/hms_multiseed/best_model_hms_seed0.pth"
