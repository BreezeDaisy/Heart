import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset_direct_outcome import CirCorDirectOutcomeDataset, direct_outcome_collate
from dataset_hms import POSITIONS
from model_direct_outcome import DirectOutcomeMultiTaskModel
from paths import CHECKPOINT_ROOT, RESULTS_ROOT, TRAIN_CSV, TRAIN_WAV_DIR


MURMUR_LABELS = {0: "Absent", 1: "Present", 2: "Unknown"}
OUTCOME_LABELS = {0: "Normal", 1: "Abnormal"}


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose direct outcome checkpoint errors.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--csv-path", type=str, default=str(TRAIN_CSV))
    parser.add_argument("--wav-dir", type=str, default=str(TRAIN_WAV_DIR))
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(CHECKPOINT_ROOT / "direct_outcome_gpu_residual_transformer_v1.pth"),
    )
    parser.add_argument("--max-segments-per-patient", type=int, default=32)
    parser.add_argument("--target-recall", type=float, default=0.88)
    parser.add_argument("--fixed-thresholds", type=str, default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_ROOT / "direct_outcome_diagnosis"))
    return parser.parse_args()


def choose_device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    specificity = float(tn / max(tn + fp, 1))
    npv = float(tn / max(tn + fn, 1))
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "abnormal_precision": float(precision),
        "abnormal_recall": float(recall),
        "abnormal_f1": float(f1),
        "specificity": specificity,
        "npv": npv,
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


def split_indices(dataset, seed):
    labels = [patient["y_outcome"] for patient in dataset.patients]
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    return train_indices, val_indices


def build_model(dataset, checkpoint_path, device):
    model = DirectOutcomeMultiTaskModel(
        clinical_dim=dataset.clinical_dim,
        num_timing_classes=dataset.num_timing_classes,
        num_grade_classes=dataset.num_grade_classes,
        num_shape_classes=dataset.num_shape_classes,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint cannot be loaded into the current model architecture. "
            "Use the model_direct_outcome.py version that created this checkpoint, "
            "or diagnose a checkpoint from the current architecture."
        ) from exc
    model.eval()
    return model


def collect_predictions(model, loader, dataset, device):
    raw_df = dataset.df.set_index("Patient ID", drop=False)
    rows = []
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            outputs = model(
                tensor_batch["x_scale1"],
                tensor_batch["x_scale2"],
                tensor_batch["x_scale3"],
                tensor_batch["position_index"],
                tensor_batch["segment_mask"],
                tensor_batch["clinical"],
                return_embedding=True,
            )
            probs = torch.softmax(outputs["outcome_logits"], dim=1)[:, 1].detach().cpu().numpy()
            attention = outputs.get("attention_weights")
            attention = attention.detach().cpu().numpy() if attention is not None else None
            position_index = batch["position_index"].cpu().numpy()
            segment_mask = batch["segment_mask"].cpu().numpy()

            for i, patient_id in enumerate(batch["patient_id"]):
                raw = raw_df.loc[str(patient_id)] if str(patient_id) in raw_df.index else {}
                y_outcome = int(batch["y_outcome"][i].item())
                y_murmur = int(batch["y_murmur"][i].item())
                valid = segment_mask[i] > 0.0
                pos_values = position_index[i][valid]
                pos_counts = Counter(POSITIONS[int(pos)] for pos in pos_values)
                attention_by_position = {f"attention_{pos}": 0.0 for pos in POSITIONS}
                top_attention_position = ""
                top_attention_weight = np.nan
                if attention is not None and valid.any():
                    weights = attention[i][valid]
                    valid_positions = pos_values.astype(int)
                    top_idx = int(np.argmax(weights))
                    top_attention_position = POSITIONS[int(valid_positions[top_idx])]
                    top_attention_weight = float(weights[top_idx])
                    for pos_idx, pos_name in enumerate(POSITIONS):
                        attention_by_position[f"attention_{pos_name}"] = float(weights[valid_positions == pos_idx].sum())

                row = {
                    "patient_id": str(patient_id),
                    "prob_abnormal": float(probs[i]),
                    "y_outcome": y_outcome,
                    "outcome": OUTCOME_LABELS.get(y_outcome, str(y_outcome)),
                    "y_murmur": y_murmur,
                    "murmur": MURMUR_LABELS.get(y_murmur, str(y_murmur)),
                    "position_count": int(batch["position_count"][i].item()),
                    "has_all_positions": bool(batch["has_all_positions"][i].item()),
                    "available_positions": ",".join(batch["available_positions"][i]),
                    "segment_count": int(valid.sum()),
                    "top_attention_position": top_attention_position,
                    "top_attention_weight": top_attention_weight,
                    "Age": raw.get("Age", ""),
                    "Sex": raw.get("Sex", ""),
                    "Height": raw.get("Height", ""),
                    "Weight": raw.get("Weight", ""),
                    "Pregnancy status": raw.get("Pregnancy status", ""),
                    "Systolic murmur timing": raw.get("Systolic murmur timing", ""),
                    "Systolic murmur grading": raw.get("Systolic murmur grading", ""),
                    "Systolic murmur shape": raw.get("Systolic murmur shape", ""),
                }
                for pos_name in POSITIONS:
                    row[f"has_{pos_name}"] = int(pos_counts.get(pos_name, 0) > 0)
                    row[f"segments_{pos_name}"] = int(pos_counts.get(pos_name, 0))
                row.update(attention_by_position)
                rows.append(row)
    return pd.DataFrame(rows)


def add_predictions_at_thresholds(df, threshold, fixed_thresholds):
    thresholds = [("selected", threshold)] + [(f"t{str(item).replace('.', '_')}", item) for item in fixed_thresholds]
    for prefix, th in thresholds:
        pred_col = f"pred_{prefix}"
        err_col = f"error_{prefix}"
        df[pred_col] = (df["prob_abnormal"] >= th).astype(int)
        df[err_col] = [error_type(int(y), int(pred)) for y, pred in zip(df["y_outcome"], df[pred_col])]
    return df


def fixed_threshold_table(df, fixed_thresholds, selected_threshold):
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
        numeric = pd.to_numeric(work[column], errors="coerce")
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
        "top_attention_position",
        "Height_missing",
        "Weight_missing",
    ]

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


def main():
    args = parse_args()
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CirCorDirectOutcomeDataset(
        csv_path=args.csv_path,
        wav_dir=args.wav_dir,
        max_segments_per_patient=args.max_segments_per_patient,
    )
    train_indices, val_indices = split_indices(dataset, args.seed)
    val_dataset = Subset(dataset, val_indices)
    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=direct_outcome_collate,
        pin_memory=device.type == "cuda",
    )

    model = build_model(dataset, checkpoint_path, device)
    pred_df = collect_predictions(model, loader, dataset, device)
    y_true = pred_df["y_outcome"].to_numpy(dtype=np.int64)
    probs = pred_df["prob_abnormal"].to_numpy(dtype=np.float32)
    selected_metrics = search_recall_threshold(y_true, probs, args.target_recall)
    selected_threshold = selected_metrics["threshold"]
    fixed_thresholds = [float(item.strip()) for item in args.fixed_thresholds.split(",") if item.strip()]
    pred_df = add_predictions_at_thresholds(pred_df, selected_threshold, fixed_thresholds)

    try:
        roc_auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        roc_auc = None
    try:
        pr_auc = float(average_precision_score(y_true, probs))
    except ValueError:
        pr_auc = None
    brier = float(brier_score_loss(y_true, probs))

    threshold_df = fixed_threshold_table(pred_df, fixed_thresholds, selected_threshold)
    group_df = group_error_summary(pred_df, selected_threshold)
    prob_df = probability_summary(pred_df)
    calib_df = calibration_table(pred_df)

    pred_path = output_dir / "patient_predictions.csv"
    threshold_path = output_dir / "threshold_metrics.csv"
    group_path = output_dir / "group_error_summary.csv"
    prob_path = output_dir / "probability_summary.csv"
    calib_path = output_dir / "calibration_table.csv"
    summary_path = output_dir / "diagnosis_summary.json"

    pred_df.sort_values("prob_abnormal", ascending=False).to_csv(pred_path, index=False, encoding="utf-8-sig")
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    group_df.to_csv(group_path, index=False, encoding="utf-8-sig")
    prob_df.to_csv(prob_path, index=False, encoding="utf-8-sig")
    calib_df.to_csv(calib_path, index=False, encoding="utf-8-sig")

    selected_error_counts = pred_df["error_selected"].value_counts().to_dict()
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "seed": args.seed,
        "val_patients": int(len(pred_df)),
        "train_patients": int(len(train_indices)),
        "target_recall": args.target_recall,
        "selected_threshold": selected_threshold,
        "selected_metrics": selected_metrics,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "selected_error_counts": selected_error_counts,
        "outcome_counts": pred_df["outcome"].value_counts().to_dict(),
        "murmur_counts": pred_df["murmur"].value_counts().to_dict(),
        "paths": {
            "patient_predictions": str(pred_path),
            "threshold_metrics": str(threshold_path),
            "group_error_summary": str(group_path),
            "probability_summary": str(prob_path),
            "calibration_table": str(calib_path),
        },
    }
    summary_path.write_text(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print("Diagnosis complete.")
    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)
    print("Brier:", round(brier, 6))
    print("Selected threshold:", round(float(selected_threshold), 4))
    print("Selected metrics:", json.dumps(to_builtin(selected_metrics), ensure_ascii=False))
    print("Selected error counts:", selected_error_counts)
    print("Wrote:", output_dir)


if __name__ == "__main__":
    main()
