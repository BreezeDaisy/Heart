import subprocess
import sys
from pathlib import Path

from final_single_config import (
    FINAL_FEATURE_CSV,
    FINAL_FEATURE_SET,
    FINAL_MODEL_PRESET,
    FINAL_OUTPUT_PREFIX,
    FINAL_THRESHOLD,
)


def main():
    script_path = Path(__file__).with_name("train_outcome_hms.py")
    default_args = [
        "--feature-csv",
        FINAL_FEATURE_CSV,
        "--feature-set",
        FINAL_FEATURE_SET,
        "--model-preset",
        FINAL_MODEL_PRESET,
        "--threshold",
        str(FINAL_THRESHOLD),
        "--output-prefix",
        FINAL_OUTPUT_PREFIX,
    ]
    command = [sys.executable, str(script_path), *default_args, *sys.argv[1:]]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
