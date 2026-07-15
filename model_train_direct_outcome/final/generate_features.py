import argparse
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import HMS_CHECKPOINT, SRC_DIR
from utils import run_command


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HMS features for the final 4-position pipeline.")
    parser.add_argument("--sample-wav-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main():
    args = parse_args()
    run_command(
        [
            sys.executable,
            SRC_DIR / "generate_hms_features_sample.py",
            "--sample-wav-dir",
            args.sample_wav_dir,
            "--checkpoint-path",
            HMS_CHECKPOINT,
            "--output-path",
            args.output_path,
            "--device",
            args.device,
        ]
    )


if __name__ == "__main__":
    main()
