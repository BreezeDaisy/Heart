import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "circor" / "training_data.csv"
DEFAULT_WAV_DIR = ROOT / "data" / "circor" / "training_data"
DEFAULT_ARCHIVE_LIST = ROOT.parent / "archive_list.txt"
POSITIONS = ["AV", "MV", "PV", "TV"]
TARGET_RECALL = 0.95

AGE_MAP = {
    "Neonate": 0.0,
    "Infant": 1.0,
    "Child": 2.0,
    "Adolescent": 3.0,
    "Young Adult": 4.0,
    "Adult": 4.0,
}
SEX_MAP = {"Female": 0.0, "Male": 1.0}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare all-patient and four-position-only evaluation sets.")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--wav-dir", type=str, default=str(DEFAULT_WAV_DIR))
    parser.add_argument("--archive-list", type=str, default=str(DEFAULT_ARCHIVE_LIST))
    parser.add_argument("--target-recall", type=float, default=TARGET_RECALL)
    return parser.parse_args()


def patient_positions_from_wav_dir(wav_dir):
    wav_dir = Path(wav_dir)
    positions = {}
    if not wav_dir.exists():
        return positions
    for wav_path in wav_dir.glob("*.wav"):
        stem = wav_path.stem
        for pos in POSITIONS:
            suffix = f"_{pos}"
            if stem.endswith(suffix):
                patient_id = stem[: -len(suffix)]
                positions.setdefault(patient_id, set()).add(pos)
                break
    return positions


def patient_positions_from_archive_list(archive_list, patient_ids):
    archive_list = Path(archive_list)
    positions = {}
    if not archive_list.exists():
        return positions
    patient_ids = set(str(pid) for pid in patient_ids)
    for line in archive_list.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.endswith(".wav"):
            continue
        name = Path(line).name[:-4]
        for pos in POSITIONS:
            suffix = f"_{pos}"
            if name.endswith(suffix):
                patient_id = name[: -len(suffix)]
                if patient_id in patient_ids:
                    positions.setdefault(patient_id, set()).add(pos)
                break
    return positions


def add_position_columns(df, wav_dir, archive_list):
    patient_ids = df["Patient ID"].astype(str).tolist()
    positions = patient_positions_from_wav_dir(wav_dir)
    source = "wav_dir"
    if not positions:
        positions = patient_positions_from_archive_list(archive_list, patient_ids)
        source = "archive_list"

    df = df.copy()
    for pos in POSITIONS:
        df[f"has_{pos}"] = df["Patient ID"].astype(str).map(lambda pid: pos in positions.get(pid, set())).astype(int)
    df["available_position_count"] = df[[f"has_{pos}" for pos in POSITIONS]].sum(axis=1)
    df["has_all_positions"] = (df["available_position_count"] == len(POSITIONS)).astype(int)
    df["position_source"] = source
    return df


def prepare_features(df):
    df = df.copy()
    df["label"] = df["Outcome"].map({"Normal": 0, "Abnormal": 1}).astype(int)
    df["age_value"] = df["Age"].map(AGE_MAP).fillna(2.0) / 4.0
    df["age_missing"] = df["Age"].isna().astype(float)
    df["sex_value"] = df["Sex"].map(SEX_MAP).fillna(0.5)
    df["sex_missing"] = df["Sex"].isna().astype(float)

    for col, denom in [("Height", 200.0), ("Weight", 120.0)]:
        numeric = pd.to_numeric(df[col], errors="coerce")
        df[f"{col.lower()}_value"] = (numeric / denom).clip(0.0, 1.5).fillna(0.0)
        df[f"{col.lower()}_missing"] = numeric.isna().astype(float)

    pregnancy_text = df["Pregnancy status"].astype(str).str.lower()
    df["pregnancy_value"] = np.where(pregnancy_text == "true", 1.0, np.where(pregnancy_text == "false", 0.0, 0.5))
    df["pregnancy_missing"] = df["Pregnancy status"].isna().astype(float)
    return df


def metrics_at_threshold(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        preds,
        labels=[1],
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "specificity": float(tn / max(tn + fp, 1)),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
    }


def choose_threshold(y_true, probs, target_recall):
    candidates = [metrics_at_threshold(y_true, probs, threshold) for threshold in np.linspace(0.05, 0.95, 91)]
    feasible = [item for item in candidates if item["recall"] >= target_recall]
    if feasible:
        return max(feasible, key=lambda item: (item["specificity"], item["accuracy"], item["threshold"]))
    return max(candidates, key=lambda item: (item["recall"], item["accuracy"]))


def run_clinical_compare(df, target_recall):
    feature_cols = [
        "age_value",
        "sex_value",
        "height_value",
        "weight_value",
        "pregnancy_value",
        "age_missing",
        "sex_missing",
        "height_missing",
        "weight_missing",
        "pregnancy_missing",
        "available_position_count",
        "has_AV",
        "has_MV",
        "has_PV",
        "has_TV",
    ]

    rows = []
    for seed in [0, 1, 2, 3, 4]:
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=seed,
            stratify=df["label"],
        )
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight={0: 1.0, 1: 3.0}, solver="liblinear"),
        )
        model.fit(train_df[feature_cols], train_df["label"])

        val_probs = model.predict_proba(val_df[feature_cols])[:, 1]
        full_metric = choose_threshold(val_df["label"].values, val_probs, target_recall)

        complete_val_df = val_df[val_df["has_all_positions"] == 1].copy()
        if len(complete_val_df) > 0 and complete_val_df["label"].nunique() == 2:
            complete_probs = model.predict_proba(complete_val_df[feature_cols])[:, 1]
            complete_metric = metrics_at_threshold(
                complete_val_df["label"].values,
                complete_probs,
                full_metric["threshold"],
            )
        else:
            complete_metric = {
                "threshold": full_metric["threshold"],
                "accuracy": np.nan,
                "recall": np.nan,
                "precision": np.nan,
                "f1": np.nan,
                "specificity": np.nan,
                "tp": np.nan,
                "fn": np.nan,
                "fp": np.nan,
                "tn": np.nan,
            }

        for split_name, metric, n_rows in [
            ("full_val", full_metric, len(val_df)),
            ("complete_position_val", complete_metric, len(complete_val_df)),
        ]:
            payload = dict(metric)
            payload["seed"] = seed
            payload["split"] = split_name
            payload["n"] = n_rows
            rows.append(payload)

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    df = pd.read_csv(args.csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["Patient ID"] = df["Patient ID"].astype(str)
    df = df[df["Outcome"].isin(["Normal", "Abnormal"])].copy()
    df = add_position_columns(df, args.wav_dir, args.archive_list)
    df = prepare_features(df)

    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    enriched_path = output_dir / "training_data_with_position_completeness.csv"
    complete_path = output_dir / "training_data_four_position_only.csv"
    df.to_csv(enriched_path, index=False, encoding="utf-8-sig")
    df[df["has_all_positions"] == 1].to_csv(complete_path, index=False, encoding="utf-8-sig")

    summary_rows = []
    for group_name, group_df in [
        ("all_patients", df),
        ("four_position_only", df[df["has_all_positions"] == 1]),
        ("missing_position", df[df["has_all_positions"] == 0]),
    ]:
        counts = group_df["Outcome"].value_counts().to_dict()
        summary_rows.append(
            {
                "group": group_name,
                "n": len(group_df),
                "normal": counts.get("Normal", 0),
                "abnormal": counts.get("Abnormal", 0),
                "abnormal_rate": counts.get("Abnormal", 0) / max(len(group_df), 1),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "position_completeness_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    compare_df = run_clinical_compare(df, args.target_recall)
    compare_path = output_dir / "position_completeness_compare_results.csv"
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")
    compare_summary = (
        compare_df.groupby("split")
        .agg(
            n_mean=("n", "mean"),
            accuracy_mean=("accuracy", "mean"),
            recall_mean=("recall", "mean"),
            specificity_mean=("specificity", "mean"),
            fn_mean=("fn", "mean"),
            fp_mean=("fp", "mean"),
            threshold_mean=("threshold", "mean"),
        )
        .reset_index()
    )
    compare_summary_path = output_dir / "position_completeness_compare_summary.csv"
    compare_summary.to_csv(compare_summary_path, index=False, encoding="utf-8-sig")

    print("Position completeness summary:")
    print(summary_df.to_string(index=False))
    print("\nClinical full-vs-complete validation comparison:")
    print(compare_summary.to_string(index=False))
    print("\nSaved:")
    print(enriched_path)
    print(complete_path)
    print(summary_path)
    print(compare_path)
    print(compare_summary_path)


if __name__ == "__main__":
    main()
