import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


MURMUR_LABELS = {0: "Absent", 1: "Present", 2: "Unknown"}
OUTCOME_LABELS = {0: "Normal", 1: "Abnormal"}


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def metrics_from_probs(y_true, probs, threshold):
    preds = (probs >= threshold).astype(np.int64)
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
        "abnormal_precision": float(precision),
        "abnormal_recall": float(recall),
        "abnormal_f1": float(f1),
        "specificity": float(tn / max(tn + fp, 1)),
        "npv": float(tn / max(tn + fn, 1)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def search_recall_threshold(y_true, probs, target_recall):
    thresholds = np.linspace(0.05, 0.95, 91)
    candidates = [metrics_from_probs(y_true, probs, threshold) for threshold in thresholds]
    feasible = [item for item in candidates if item["abnormal_recall"] >= target_recall]
    if feasible:
        return max(feasible, key=lambda item: (item["specificity"], item["accuracy"], item["threshold"]))
    return max(candidates, key=lambda item: (item["abnormal_recall"], item["accuracy"], item["threshold"]))


def error_type(y_true, pred):
    if y_true == 1 and pred == 1:
        return "TP"
    if y_true == 0 and pred == 0:
        return "TN"
    if y_true == 0 and pred == 1:
        return "FP"
    return "FN"


def add_threshold_predictions(df, selected_threshold, fixed_thresholds):
    thresholds = [("selected", selected_threshold)] + [(f"t{str(th).replace('.', '_')}", th) for th in fixed_thresholds]
    for prefix, threshold in thresholds:
        pred_col = f"pred_{prefix}"
        err_col = f"error_{prefix}"
        df[pred_col] = (df["prob_abnormal"] >= threshold).astype(int)
        df[err_col] = [error_type(int(y), int(pred)) for y, pred in zip(df["y_outcome"], df[pred_col])]
    return df


def threshold_metrics_table(df, fixed_thresholds, selected_threshold):
    y_true = df["y_outcome"].to_numpy(dtype=np.int64)
    probs = df["prob_abnormal"].to_numpy(dtype=np.float32)
    rows = []
    for name, threshold in [("selected", selected_threshold)] + [(f"fixed_{th:.2f}", th) for th in fixed_thresholds]:
        item = metrics_from_probs(y_true, probs, threshold)
        item["threshold_name"] = name
        item["subset"] = "full_val"
        rows.append(item)

        complete = df[df["has_all_positions"]]
        if len(complete) > 0:
            item = metrics_from_probs(
                complete["y_outcome"].to_numpy(dtype=np.int64),
                complete["prob_abnormal"].to_numpy(dtype=np.float32),
                threshold,
            )
            item["threshold_name"] = name
            item["subset"] = "complete_position_val"
            rows.append(item)
    return pd.DataFrame(rows)


def group_error_summary(df, selected_threshold):
    work = df.copy()
    work["pred"] = (work["prob_abnormal"] >= selected_threshold).astype(int)
    work["error"] = [error_type(int(y), int(pred)) for y, pred in zip(work["y_outcome"], work["pred"])]
    for column in ["Height", "Weight"]:
        numeric = pd.to_numeric(work.get(column, pd.Series(dtype=float)), errors="coerce")
        work[f"{column}_missing"] = np.where(numeric.isna(), "missing", "present")
    group_columns = [
        "murmur",
        "outcome",
        "Age",
        "Sex",
        "Pregnancy status",
        "position_count",
        "has_all_positions",
        "available_positions",
        "Height_missing",
        "Weight_missing",
    ]
    if "top_attention_position" in work.columns:
        group_columns.append("top_attention_position")

    rows = []
    for column in group_columns:
        if column not in work.columns:
            continue
        for value, group in work.groupby(column, dropna=False):
            counts = group["error"].value_counts()
            total = int(len(group))
            abnormal_total = int((group["y_outcome"] == 1).sum())
            normal_total = int((group["y_outcome"] == 0).sum())
            fp = int(counts.get("FP", 0))
            fn = int(counts.get("FN", 0))
            rows.append(
                {
                    "group_by": column,
                    "group_value": str(value),
                    "total": total,
                    "abnormal_total": abnormal_total,
                    "normal_total": normal_total,
                    "TP": int(counts.get("TP", 0)),
                    "TN": int(counts.get("TN", 0)),
                    "FP": fp,
                    "FN": fn,
                    "fp_rate_among_normals": float(fp / max(normal_total, 1)),
                    "fn_rate_among_abnormals": float(fn / max(abnormal_total, 1)),
                    "mean_prob_abnormal": float(group["prob_abnormal"].mean()),
                    "median_prob_abnormal": float(group["prob_abnormal"].median()),
                }
            )
    return pd.DataFrame(rows).sort_values(["group_by", "total"], ascending=[True, False])


def probability_summary(df):
    rows = []
    for name, group in df.groupby("outcome"):
        probs = group["prob_abnormal"]
        rows.append(
            {
                "group": name,
                "count": int(len(group)),
                "mean": float(probs.mean()),
                "std": float(probs.std(ddof=0)),
                "min": float(probs.min()),
                "p10": float(probs.quantile(0.10)),
                "p25": float(probs.quantile(0.25)),
                "median": float(probs.quantile(0.50)),
                "p75": float(probs.quantile(0.75)),
                "p90": float(probs.quantile(0.90)),
                "max": float(probs.max()),
            }
        )
    return pd.DataFrame(rows)


def calibration_table(df, bins=10):
    work = df.copy()
    work["bin"] = pd.cut(work["prob_abnormal"], bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True)
    rows = []
    for bin_value, group in work.groupby("bin", observed=False):
        if len(group) == 0:
            continue
        rows.append(
            {
                "prob_bin": str(bin_value),
                "count": int(len(group)),
                "mean_prob_abnormal": float(group["prob_abnormal"].mean()),
                "observed_abnormal_rate": float(group["y_outcome"].mean()),
            }
        )
    return pd.DataFrame(rows)


def subgroup_auc_summary(df):
    rows = []
    groups = {
        "full": df,
        "complete_position": df[df["has_all_positions"] == True],
        "incomplete_position": df[df["has_all_positions"] == False],
        "murmur_absent": df[df["murmur"] == "Absent"],
        "murmur_present": df[df["murmur"] == "Present"],
        "murmur_unknown": df[df["murmur"] == "Unknown"],
    }
    for name, group in groups.items():
        if len(group) == 0 or group["y_outcome"].nunique() < 2:
            continue
        y_true = group["y_outcome"].to_numpy(dtype=np.int64)
        probs = group["prob_abnormal"].to_numpy(dtype=np.float32)
        rows.append(
            {
                "subset": name,
                "count": int(len(group)),
                "abnormal_count": int(y_true.sum()),
                "normal_count": int(len(group) - y_true.sum()),
                "roc_auc": float(roc_auc_score(y_true, probs)),
                "pr_auc": float(average_precision_score(y_true, probs)),
                "normal_mean_prob": float(group[group["y_outcome"] == 0]["prob_abnormal"].mean()),
                "abnormal_mean_prob": float(group[group["y_outcome"] == 1]["prob_abnormal"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_diagnostics(pred_df, output_dir, target_recall=0.88, fixed_thresholds=None, extra_summary=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_thresholds = fixed_thresholds or [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    y_true = pred_df["y_outcome"].to_numpy(dtype=np.int64)
    probs = pred_df["prob_abnormal"].to_numpy(dtype=np.float32)
    selected_metrics = search_recall_threshold(y_true, probs, target_recall)
    selected_threshold = selected_metrics["threshold"]
    pred_df = add_threshold_predictions(pred_df.copy(), selected_threshold, fixed_thresholds)

    roc_auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) == 2 else None
    pr_auc = float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) == 2 else None
    brier = float(brier_score_loss(y_true, probs))

    paths = {
        "patient_predictions": output_dir / "patient_predictions.csv",
        "threshold_metrics": output_dir / "threshold_metrics.csv",
        "group_error_summary": output_dir / "group_error_summary.csv",
        "probability_summary": output_dir / "probability_summary.csv",
        "calibration_table": output_dir / "calibration_table.csv",
        "subgroup_auc_summary": output_dir / "subgroup_auc_summary.csv",
        "diagnosis_summary": output_dir / "diagnosis_summary.json",
    }
    pred_df.sort_values("prob_abnormal", ascending=False).to_csv(
        paths["patient_predictions"], index=False, encoding="utf-8-sig"
    )
    threshold_metrics_table(pred_df, fixed_thresholds, selected_threshold).to_csv(
        paths["threshold_metrics"], index=False, encoding="utf-8-sig"
    )
    group_error_summary(pred_df, selected_threshold).to_csv(
        paths["group_error_summary"], index=False, encoding="utf-8-sig"
    )
    probability_summary(pred_df).to_csv(paths["probability_summary"], index=False, encoding="utf-8-sig")
    calibration_table(pred_df).to_csv(paths["calibration_table"], index=False, encoding="utf-8-sig")
    subgroup_auc_summary(pred_df).to_csv(paths["subgroup_auc_summary"], index=False, encoding="utf-8-sig")

    summary = {
        "target_recall": float(target_recall),
        "selected_threshold": selected_threshold,
        "selected_metrics": selected_metrics,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "selected_error_counts": pred_df["error_selected"].value_counts().to_dict(),
        "outcome_counts": pred_df["outcome"].value_counts().to_dict(),
        "murmur_counts": pred_df["murmur"].value_counts().to_dict(),
        "paths": {key: str(value) for key, value in paths.items() if key != "diagnosis_summary"},
    }
    if extra_summary:
        summary.update(extra_summary)
    paths["diagnosis_summary"].write_text(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
