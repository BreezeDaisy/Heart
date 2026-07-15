import subprocess
import sys
from pathlib import Path


def main():
    script_path = Path(__file__).resolve().parent / "src" / "train_direct_outcome.py"
    default_args = [
        "--seed",
        "0",
        "--device",
        "cuda",
        "--epochs",
        "35",
        "--patience",
        "8",
        "--batch-size",
        "8",
        "--num-workers",
        "0",
        "--target-recall",
        "0.93",
        "--complete-target-recall-delta",
        "0.03",
        "--absent-abnormal-target-recall-delta",
        "0.03",
        "--outcome-abnormal-weight",
        "1.8",
        "--absent-abnormal-weight",
        "1.5",
        "--fn-penalty-weight",
        "0.10",
        "--alpha-murmur",
        "0.00",
        "--alpha-timing",
        "0.03",
        "--alpha-grade",
        "0.03",
        "--alpha-shape",
        "0.02",
        "--checkpoint-path",
        str(Path(__file__).resolve().parent / "checkpoints" / "direct_outcome_murmur_stratified_v1.pth"),
    ]
    command = [sys.executable, str(script_path), *default_args, *sys.argv[1:]]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
