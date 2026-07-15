from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "circor" / "training_data.csv"
TARGET_RECALL = 0.95
SEEDS = [0, 1, 2, 3, 4]


AGE_MAP = {
    "Neonate": 0.0,
    "Infant": 1.0,
    "Child": 2.0,
    "Adolescent": 3.0,
    "Young Adult": 4.0,
    "Adult": 4.0,
}
SEX_MAP = {"Female": 0.0, "Male": 1.0}


def prepare_df():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Outcome"].isin(["Normal", "Abnormal"])].copy()
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

    # Analysis-only feature: not recommended as an inference input. It is useful
    # here to show how much of the outcome signal is locked behind murmur labels.
    df["murmur_present"] = (df["Murmur"] == "Present").astype(float)
    df["murmur_unknown"] = (df["Murmur"] == "Unknown").astype(float)
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
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, preds),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "specificity": tn / max(tn + fp, 1),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


def choose_threshold(y_true, probs):
    candidates = [metrics_at_threshold(y_true, probs, threshold) for threshold in np.linspace(0.05, 0.95, 91)]
    feasible = [item for item in candidates if item["recall"] >= TARGET_RECALL]
    if feasible:
        return max(feasible, key=lambda item: (item["specificity"], item["accuracy"], item["threshold"]))
    return max(candidates, key=lambda item: (item["recall"], item["accuracy"]))


def run_feature_set(df, name, columns):
    rows = []
    for seed in SEEDS:
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
        model.fit(train_df[columns], train_df["label"])
        probs = model.predict_proba(val_df[columns])[:, 1]
        metrics = choose_threshold(val_df["label"].values, probs)
        metrics["seed"] = seed
        metrics["feature_set"] = name
        rows.append(metrics)
    return pd.DataFrame(rows)


def main():
    df = prepare_df()
    feature_sets = {
        "age_sex": [
            "age_value",
            "sex_value",
            "age_missing",
            "sex_missing",
        ],
        "clinical_added": [
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
        ],
        "clinical_plus_murmur_oracle": [
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
            "murmur_present",
            "murmur_unknown",
        ],
    }

    results = []
    for name, columns in feature_sets.items():
        results.append(run_feature_set(df, name, columns))

    out_df = pd.concat(results, ignore_index=True)
    summary = (
        out_df.groupby("feature_set")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            recall_mean=("recall", "mean"),
            specificity_mean=("specificity", "mean"),
            precision_mean=("precision", "mean"),
            fn_mean=("fn", "mean"),
            fp_mean=("fp", "mean"),
            threshold_mean=("threshold", "mean"),
        )
        .reset_index()
    )

    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    out_df.to_csv(output_dir / "clinical_feature_ablation_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "clinical_feature_ablation_summary.csv", index=False, encoding="utf-8-sig")

    print("Per-seed results:")
    print(out_df.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
