import subprocess
import sys
from pathlib import Path

from final_single_config import FINAL_FEATURE_CSV, FINAL_FEATURE_SET, FINAL_MODEL_PRESET, FINAL_THRESHOLD


def main():
    script_path = Path(__file__).with_name("run_outcome_timegrade_inference.py")
    default_args = [
        "--feature-csv",
        FINAL_FEATURE_CSV,
        "--feature-set",
        FINAL_FEATURE_SET,
        "--model-preset",
        FINAL_MODEL_PRESET,
        "--threshold",
        str(FINAL_THRESHOLD),
    ]
    command = [sys.executable, str(script_path), *default_args, *sys.argv[1:]]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
