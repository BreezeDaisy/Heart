import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from baseline_utils import MURMUR_LABELS, OUTCOME_LABELS, search_recall_threshold
from dataset_audio_primary import CirCorAudioPrimaryDataset, audio_primary_collate
from dataset_hms import POSITIONS
from paths import CHECKPOINT_ROOT, RESULTS_ROOT, TRAIN_CSV, TRAIN_WAV_DIR
from train_audio_primary import AudioPrimaryOutcomeModel, choose_device, move_batch


ACOUSTIC_FEATURE_NAMES = [
    "rms",
    "abs_mean",
    "peak",
    "crest",
    "zcr",
    "centroid",
    "bandwidth",
    "rolloff",
    "flatness",
    "spectral_entropy",
    "low_ratio",
    "mid_minus_high_ratio",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Export patient and segment diagnostics for an audio-primary model.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--segment-duration", type=float, default=3.0)
    parser.add_argument("--segment-hop", type=float, default=2.0)
    parser.add_argument("--max-segments-per-patient", type=int, default=32)
    parser.add_argument("--target-recall", type=float, default=0.88)
    parser.add_argument("--selected-threshold", type=float, default=None)
    parser.add_argument("--segment-evidence-threshold", type=float, default=0.50)
    parser.add_argument("--csv-path", type=str, default=str(TRAIN_CSV))
    parser.add_argument("--wav-dir", type=str, default=str(TRAIN_WAV_DIR))
    parser.add_argument("--checkpoint-path", type=str, default=str(CHECKPOINT_ROOT / "audio_primary_v3.pth"))
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_ROOT / "audio_primary_v3_supplement"))
    return parser.parse_args()


def load_model(checkpoint_path, dataset, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    position_dropout = float(saved_args.get("position_embedding_dropout", 0.0))
    model = AudioPrimaryOutcomeModel(
        acoustic_dim=dataset.acoustic_dim,
        weak_clinical_dim=dataset.weak_clinical_dim,
        position_embedding_dropout=position_dropout,
    ).to(device)
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def entropy(values):
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total <= 1e-12 or len(values) <= 1:
        return 0.0
    probs = values / total
    return float(-(probs * np.log(probs + 1e-12)).sum() / math.log(len(values)))


def topk_mean(values, k):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.nan
    k = min(k, values.size)
    return float(np.sort(values)[-k:].mean())


def auc_or_none(y_true, probs):
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    if len(np.unique(y_true)) < 2:
        return None, None
    return float(roc_auc_score(y_true, probs)), float(average_precision_score(y_true, probs))


def threshold_tradeoff(y_true, probs, max_fn_values=(5, 7, 10), max_fp_values=(10, 20)):
    rows = []
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    thresholds = sorted(set(np.round(np.linspace(0.0, 1.0, 101), 4).tolist() + probs.tolist()))
    for threshold in thresholds:
        pred = probs >= threshold
        fn = int(np.sum((y_true == 1) & ~pred))
        fp = int(np.sum((y_true == 0) & pred))
        tp = int(np.sum((y_true == 1) & pred))
        tn = int(np.sum((y_true == 0) & ~pred))
        rows.append({"threshold": float(threshold), "fn": fn, "fp": fp, "tp": tp, "tn": tn})
    df = pd.DataFrame(rows)

    summary = []
    for max_fn in max_fn_values:
        feasible = df[df["fn"] <= max_fn]
        if not feasible.empty:
            row = feasible.sort_values(["fp", "fn", "threshold"], ascending=[True, True, False]).iloc[0].to_dict()
            summary.append({"constraint": f"fn<={max_fn}", **row})
    for max_fp in max_fp_values:
        feasible = df[df["fp"] <= max_fp]
        if not feasible.empty:
            row = feasible.sort_values(["fn", "fp", "threshold"], ascending=[True, True, True]).iloc[0].to_dict()
            summary.append({"constraint": f"fp<={max_fp}", **row})
    return df, pd.DataFrame(summary)


def export_predictions(model, loader, dataset, device, args):
    raw_df = dataset.df.set_index("Patient ID", drop=False)
    patient_rows = []
    segment_rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = move_batch(batch, device)
            outputs = model(
                moved["x_scale1"],
                moved["x_scale2"],
                moved["x_scale3"],
                moved["acoustic"],
                moved["position_index"],
                moved["segment_mask"],
                moved["weak_clinical"],
                return_embedding=True,
            )
            patient_probs = torch.softmax(outputs["outcome_logits"], dim=1)[:, 1].detach().cpu().numpy()
            segment_probs = torch.softmax(outputs["segment_outcome_logits"], dim=-1)[..., 1].detach().cpu().numpy()
            attention = outputs["attention_weights"].detach().cpu().numpy()
            position_index = batch["position_index"].cpu().numpy()
            starts = batch["segment_start_sec"].cpu().numpy()
            mask = batch["segment_mask"].cpu().numpy()
            acoustic = batch["acoustic"].cpu().numpy()

            for i, patient_id in enumerate(batch["patient_id"]):
                raw = raw_df.loc[str(patient_id)] if str(patient_id) in raw_df.index else {}
                y_outcome = int(batch["y_outcome"][i].item())
                y_murmur = int(batch["y_murmur"][i].item())
                valid = mask[i] > 0.0
                pos_values = position_index[i][valid]
                start_values = starts[i][valid]
                seg_probs = segment_probs[i][valid]
                weights = attention[i][valid]
                acoustic_values = acoustic[i][valid]
                selected_pred = int(patient_probs[i] >= args.selected_threshold) if args.selected_threshold is not None else -1

                attention_by_position = {}
                prob_mean_by_position = {}
                prob_max_by_position = {}
                evidence_by_position = {}
                segments_by_position = {}
                for pos_idx, pos_name in enumerate(POSITIONS):
                    pos_mask = pos_values == pos_idx
                    pos_probs = seg_probs[pos_mask]
                    attention_by_position[f"attention_{pos_name}"] = float(weights[pos_mask].sum()) if pos_mask.any() else 0.0
                    prob_mean_by_position[f"segment_prob_mean_{pos_name}"] = float(pos_probs.mean()) if pos_probs.size else np.nan
                    prob_max_by_position[f"segment_prob_max_{pos_name}"] = float(pos_probs.max()) if pos_probs.size else np.nan
                    evidence_by_position[f"evidence_{pos_name}"] = int(np.sum(pos_probs >= args.segment_evidence_threshold))
                    segments_by_position[f"segments_{pos_name}"] = int(np.sum(pos_mask))

                evidence_mask = seg_probs >= args.segment_evidence_threshold
                row = {
                    "patient_id": str(patient_id),
                    "prob_abnormal": float(patient_probs[i]),
                    "selected_threshold": args.selected_threshold,
                    "pred_selected": selected_pred,
                    "error_selected": (
                        "TP" if selected_pred == 1 and y_outcome == 1 else
                        "TN" if selected_pred == 0 and y_outcome == 0 else
                        "FP" if selected_pred == 1 and y_outcome == 0 else
                        "FN" if selected_pred == 0 and y_outcome == 1 else
                        ""
                    ),
                    "y_outcome": y_outcome,
                    "outcome": OUTCOME_LABELS.get(y_outcome, str(y_outcome)),
                    "y_murmur": y_murmur,
                    "murmur": MURMUR_LABELS.get(y_murmur, str(y_murmur)),
                    "position_count": int(batch["position_count"][i].item()),
                    "has_all_positions": bool(batch["has_all_positions"][i].item()),
                    "available_positions": ",".join(batch["available_positions"][i]),
                    "segment_count": int(valid.sum()),
                    "segment_mean_prob": float(seg_probs.mean()) if seg_probs.size else np.nan,
                    "segment_max_prob": float(seg_probs.max()) if seg_probs.size else np.nan,
                    "segment_top3_mean_prob": topk_mean(seg_probs, 3),
                    "segment_top5_mean_prob": topk_mean(seg_probs, 5),
                    "segment_prob_std": float(seg_probs.std()) if seg_probs.size else np.nan,
                    "segment_prob_entropy": entropy(seg_probs) if seg_probs.size else np.nan,
                    "attention_entropy": entropy(weights) if weights.size else np.nan,
                    "evidence_segment_count": int(evidence_mask.sum()),
                    "evidence_segment_ratio": float(evidence_mask.mean()) if evidence_mask.size else np.nan,
                    "evidence_position_count": int(
                        sum(1 for pos_name in POSITIONS if evidence_by_position[f"evidence_{pos_name}"] > 0)
                    ),
                    "top_attention_position": POSITIONS[int(pos_values[int(np.argmax(weights))])] if weights.size else "",
                    "top_attention_weight": float(weights.max()) if weights.size else np.nan,
                    "Age": raw.get("Age", ""),
                    "Sex": raw.get("Sex", ""),
                    "Height": raw.get("Height", ""),
                    "Weight": raw.get("Weight", ""),
                    "Pregnancy status": raw.get("Pregnancy status", ""),
                }
                for pos_name in POSITIONS:
                    row[f"has_{pos_name}"] = int(pos_name in batch["available_positions"][i])
                row.update(segments_by_position)
                row.update(attention_by_position)
                row.update(prob_mean_by_position)
                row.update(prob_max_by_position)
                row.update(evidence_by_position)
                patient_rows.append(row)

                for segment_idx, (pos_idx, start_sec, prob, attn, acoustic_features) in enumerate(
                    zip(pos_values, start_values, seg_probs, weights, acoustic_values)
                ):
                    seg_row = {
                        "patient_id": str(patient_id),
                        "segment_index": segment_idx,
                        "position": POSITIONS[int(pos_idx)],
                        "start_sec": float(start_sec),
                        "end_sec": float(start_sec + args.segment_duration),
                        "segment_prob_abnormal": float(prob),
                        "attention_weight": float(attn),
                        "is_evidence_segment": int(prob >= args.segment_evidence_threshold),
                        "patient_prob_abnormal": float(patient_probs[i]),
                        "y_outcome": y_outcome,
                        "outcome": OUTCOME_LABELS.get(y_outcome, str(y_outcome)),
                        "y_murmur": y_murmur,
                        "murmur": MURMUR_LABELS.get(y_murmur, str(y_murmur)),
                    }
                    for name, value in zip(ACOUSTIC_FEATURE_NAMES, acoustic_features):
                        seg_row[f"acoustic_{name}"] = float(value)
                    segment_rows.append(seg_row)

    return pd.DataFrame(patient_rows), pd.DataFrame(segment_rows)


def write_summary(patient_df, segment_df, checkpoint, output_dir, args):
    y_true = patient_df["y_outcome"].to_numpy(dtype=np.int64)
    probs = patient_df["prob_abnormal"].to_numpy(dtype=np.float32)
    roc_auc, pr_auc = auc_or_none(y_true, probs)
    selected = search_recall_threshold(y_true, probs, args.target_recall)
    threshold_for_report = args.selected_threshold if args.selected_threshold is not None else selected["threshold"]

    subset_rows = []
    for name, group in [
        ("full", patient_df),
        ("murmur_absent", patient_df[patient_df["murmur"] == "Absent"]),
        ("murmur_present", patient_df[patient_df["murmur"] == "Present"]),
        ("murmur_unknown", patient_df[patient_df["murmur"] == "Unknown"]),
        ("complete_position", patient_df[patient_df["has_all_positions"] == True]),
        ("incomplete_position", patient_df[patient_df["has_all_positions"] == False]),
    ]:
        if group.empty:
            continue
        subset_auc, subset_pr = auc_or_none(group["y_outcome"], group["prob_abnormal"])
        subset_rows.append(
            {
                "subset": name,
                "count": int(len(group)),
                "abnormal_count": int((group["y_outcome"] == 1).sum()),
                "normal_count": int((group["y_outcome"] == 0).sum()),
                "roc_auc": subset_auc,
                "pr_auc": subset_pr,
                "normal_mean_prob": float(group.loc[group["y_outcome"] == 0, "prob_abnormal"].mean()),
                "abnormal_mean_prob": float(group.loc[group["y_outcome"] == 1, "prob_abnormal"].mean()),
            }
        )
    subset_df = pd.DataFrame(subset_rows)

    absent = patient_df[patient_df["murmur"] == "Absent"].copy()
    absent_thresholds, absent_constraints = threshold_tradeoff(absent["y_outcome"], absent["prob_abnormal"])
    absent_thresholds.to_csv(output_dir / "murmur_absent_threshold_tradeoff.csv", index=False, encoding="utf-8-sig")
    absent_constraints.to_csv(output_dir / "murmur_absent_constraint_summary.csv", index=False, encoding="utf-8-sig")
    subset_df.to_csv(output_dir / "subgroup_auc_summary.csv", index=False, encoding="utf-8-sig")

    position_duration = (
        segment_df.groupby(["patient_id", "position"], as_index=False)
        .agg(
            segment_count=("segment_index", "count"),
            first_start_sec=("start_sec", "min"),
            last_end_sec=("end_sec", "max"),
            mean_segment_prob=("segment_prob_abnormal", "mean"),
            max_segment_prob=("segment_prob_abnormal", "max"),
            attention_sum=("attention_weight", "sum"),
            evidence_segment_count=("is_evidence_segment", "sum"),
        )
    )
    position_duration.to_csv(output_dir / "position_segment_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "checkpoint_path": str(args.checkpoint_path),
        "output_dir": str(output_dir),
        "patient_count": int(len(patient_df)),
        "segment_count": int(len(segment_df)),
        "outcome_counts": dict(Counter(patient_df["outcome"])),
        "murmur_counts": dict(Counter(patient_df["murmur"])),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "target_recall_threshold": selected,
        "report_threshold": threshold_for_report,
        "checkpoint_best_metric": checkpoint.get("best_metric") if isinstance(checkpoint, dict) else None,
        "checkpoint_best_epoch": checkpoint.get("best_epoch") if isinstance(checkpoint, dict) else None,
        "files": {
            "patient_level": str(output_dir / "patient_level_diagnostics.csv"),
            "segment_level": str(output_dir / "segment_level_diagnostics.csv"),
            "position_segment_summary": str(output_dir / "position_segment_summary.csv"),
            "subgroup_auc_summary": str(output_dir / "subgroup_auc_summary.csv"),
            "murmur_absent_threshold_tradeoff": str(output_dir / "murmur_absent_threshold_tradeoff.csv"),
            "murmur_absent_constraint_summary": str(output_dir / "murmur_absent_constraint_summary.csv"),
        },
    }
    (output_dir / "diagnostic_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main():
    args = parse_args()
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CirCorAudioPrimaryDataset(
        csv_path=args.csv_path,
        wav_dir=args.wav_dir,
        segment_duration=args.segment_duration,
        segment_hop=args.segment_hop,
        max_segments_per_patient=args.max_segments_per_patient,
        training=False,
    )
    indices = list(range(len(dataset)))
    labels = [dataset.patients[index]["y_outcome"] for index in indices]
    _, val_indices = train_test_split(indices, test_size=0.2, random_state=args.seed, stratify=labels)
    loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=audio_primary_collate,
        pin_memory=device.type == "cuda",
    )
    model, checkpoint = load_model(args.checkpoint_path, dataset, device)
    if args.selected_threshold is None and isinstance(checkpoint, dict):
        best_metric = checkpoint.get("best_metric") or {}
        args.selected_threshold = best_metric.get("threshold")

    patient_df, segment_df = export_predictions(model, loader, dataset, device, args)
    patient_df.to_csv(output_dir / "patient_level_diagnostics.csv", index=False, encoding="utf-8-sig")
    segment_df.to_csv(output_dir / "segment_level_diagnostics.csv", index=False, encoding="utf-8-sig")
    summary = write_summary(patient_df, segment_df, checkpoint, output_dir, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
