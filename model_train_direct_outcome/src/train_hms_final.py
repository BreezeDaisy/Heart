import subprocess
import sys
from pathlib import Path

from final_single_config import FINAL_TRAIN_ARGS


def main():
    script_path = Path(__file__).with_name("train_hms.py")
    command = [sys.executable, str(script_path), *FINAL_TRAIN_ARGS, *sys.argv[1:]]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
