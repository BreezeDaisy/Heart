import subprocess
import sys
from pathlib import Path

from final_single_config import FINAL_FEATURE_CSV, FINAL_HMS_CHECKPOINT


def main():
    script_path = Path(__file__).with_name("generate_hms_features.py")
    default_args = [
        "--checkpoint-path",
        FINAL_HMS_CHECKPOINT,
        "--output-path",
        FINAL_FEATURE_CSV,
        "--batch-size",
        "32",
        "--num-workers",
        "0",
        "--device",
        "cuda",
    ]
    command = [sys.executable, str(script_path), *default_args, *sys.argv[1:]]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
