from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "circor"
TRAIN_CSV = DATA_ROOT / "training_data.csv"
TRAIN_WAV_DIR = DATA_ROOT / "training_data"
CHECKPOINT_DIR = ROOT / "checkpoints"


def main():
    print("Repro package root:", ROOT)

    checks = [
        ("training_data.csv", TRAIN_CSV.exists(), TRAIN_CSV),
        ("training_data wav dir", TRAIN_WAV_DIR.exists(), TRAIN_WAV_DIR),
        ("checkpoints dir", CHECKPOINT_DIR.exists(), CHECKPOINT_DIR),
        ("requirements.txt", (ROOT / "requirements.txt").exists(), ROOT / "requirements.txt"),
        ("train launcher", (ROOT / "train_hms_final.py").exists(), ROOT / "train_hms_final.py"),
        ("feature launcher", (ROOT / "generate_hms_features_final.py").exists(), ROOT / "generate_hms_features_final.py"),
        ("outcome train launcher", (ROOT / "train_outcome_hms_final.py").exists(), ROOT / "train_outcome_hms_final.py"),
        ("outcome inference launcher", (ROOT / "run_outcome_final_inference.py").exists(), ROOT / "run_outcome_final_inference.py"),
        ("full reproduce launcher", (ROOT / "reproduce_full_outcome.py").exists(), ROOT / "reproduce_full_outcome.py"),
    ]

    missing = []
    for label, ok, path in checks:
        status = "OK" if ok else "MISSING"
        print(f"[{status}] {label}: {path}")
        if not ok:
            missing.append(label)

    if TRAIN_WAV_DIR.exists():
        wav_count = sum(1 for _ in TRAIN_WAV_DIR.rglob("*.wav"))
        print("WAV count:", wav_count)
        if wav_count == 0:
            missing.append("wav files")

    if missing:
        print("\nPackage is not ready yet.")
        print("Fill the missing items above, then rerun this check.")
        raise SystemExit(1)

    print("\nPackage is ready for HMS retraining.")
    print("Train with: python train_hms_final.py")
    print("Regenerate features with: python generate_hms_features_final.py")
    print("Train outcome with: python train_outcome_hms_final.py")
    print("Run binary outcome inference with: python run_outcome_final_inference.py")
    print("Run end-to-end with: python reproduce_full_outcome.py")


if __name__ == "__main__":
    main()
